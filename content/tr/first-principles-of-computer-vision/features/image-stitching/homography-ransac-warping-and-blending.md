# Homografi Hesabı, RANSAC, Görüntü Eğme ve Harmanlama (Homography Estimation, RANSAC, Warping and Blending)

<!-- toc -->

## 1. Homografi Hesaplama (Computing Homography)

### 1.1 Görüntü Birleştirmedeki Rolü

Bir kamerayı kendi optik merkezi etrafında döndürerek farklı açılardan görüntüler kaydettiğimizde, elde edilen tüm görüntü düzlemleri (örneğin $\Pi_1, \Pi_2, \Pi_3$) aynı projeksiyon merkezini paylaştıkları için birbirlerine doğrudan birer homografi matrisiyle bağlıdırlar. Bu homografiler bileşke kuralı ile birbirleriyle çarpılarak tüm görüntüler tek bir referans düzlemine ($\Pi_p$) kusursuzca hizalanabilir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-01.png" alt="Ortak Projeksiyon Merkezinden Çekilen Görüntü Düzlemleri" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 1: Pinhole etrafında dönen kameranın görüntü düzlemleri (Π₁, Π₂, Π₃) ve ortak referans düzlemine (Πₚ) homografik izdüşümü.</em></figcaption>
  </div>
</figure>

### 1.2 Homografinin Geçerlilik Koşulları

Homografi ile görüntü hizalamanın matematiksel olarak geçerli olduğu üç temel durum mevcuttur:

1. **Aynı Bakış Açısı (Same Viewpoint):** Kameranın sadece kendi optik merkezi etrafında döndürüldüğü (saf rotasyon), yani bakış açısının kesinlikle değişmediği durumlar. Bu durumda 3B sahnenin derinlik karmaşıklığı ne olursa olsun homografi her zaman kusursuz çalışır.
2. **Düzlemsel Sahneler (Planar Scene):** Kamera farklı konumlara ötelenip hareket etse dahi, fotoğraflanan nesnenin kendisi 3B uzayda tamamen düzlemsel (*planar*) bir yapıya sahipse (örneğin duvardaki tablo veya bina cephesi) homografi yine de tamamen geçerlidir.
3. **Sonsuzdaki Düzlem (Plane at Infinity):** Kamera hareket etse bile sahne kameraya kıyasla çok uzaktaysa (örneğin manzara çekimleri), sahne sonsuzdaki tek bir düzlem olarak kabul edilebilir ve homografi geçerliliğini korur.

> **Geçersiz Durum (Paralaks Etkisi):** Sahnenin kameraya yakın olduğu, ciddi 3B derinlik varyasyonları içerdiği ve kameranın ötelenerek hareket ettirildiği durumlarda homografi geçerliliğini yitirir ve görüntü hizalamada kırılmalar (*parallax artifacts*) oluşur.

### 1.3 Matematiksel Çözüm (Direct Linear Transform - DLT)

Kaynak (*source*) görüntüdeki bir $p_s[x_s, y_s, 1]^T$ noktasını, hedef (*destination*) görüntüdeki $p_d[x_d, y_d, 1]^T$ noktasına haritalayan 3x3'lük $H$ homografi matrisini hesaplayalım:

$$p_d \equiv H \cdot p_s$$

$$\begin{bmatrix} \tilde{x}_d \\ \tilde{y}_d \\ \tilde{z}_d \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_s \\ y_s \\ 1 \end{bmatrix}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-02.png" alt="Eşleşen Noktalar Üzerinden Homografi Dönüşümü" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 2: Kaynak görüntüdeki (Source) nokta pₛ ile hedef görüntüdeki (Destination) nokta p_d arasındaki homografi eşlemesi.</em></figcaption>
  </div>
</figure>

Bu doğrusal sistemi açık yazıp homojen normalizasyonu gerçekleştirdiğimizde ($x_d = \tilde{x}_d / \tilde{z}_d$ ve $y_d = \tilde{y}_d / \tilde{z}_d$), her bir karşılık gelen nokta çifti için şu iki temel denklemi elde ederiz:

$$x_d = \frac{h_{11}x_s + h_{12}y_s + h_{13}}{h_{31}x_s + h_{32}y_s + h_{33}}$$

$$y_d = \frac{h_{21}x_s + h_{22}y_s + h_{23}}{h_{31}x_s + h_{32}y_s + h_{33}}$$

Denklemleri paydalardan kurtarıp bilinmeyen $h_{ij}$ parametrelerine göre düzenlersek:

$$x_s h_{11} + y_s h_{12} + h_{13} - x_d x_s h_{31} - x_d y_s h_{32} - x_d h_{33} = 0$$

$$x_s h_{21} + y_s h_{22} + h_{23} - y_d x_s h_{31} - y_d y_s h_{32} - y_d h_{33} = 0$$

Görüldüğü üzere, her bir eşleşen nokta çifti bize 2 adet bağımsız doğrusal denklem sağlar. Homografinin 8 serbestlik derecesini çözebilmek için **en az 4 çift eşleşen noktaya (minimum 4 pairs)** ihtiyacımız vardır.

### 1.4 Kısıtlı En Küçük Kareler (Constrained Least Squares)

Pratikte ölçüm gürültülerini azaltmak için $4$'ten fazla ($N$ adet) nokta çifti kullanılır. Bu durumda karşımıza aşırı belirlenmiş (*overdetermined*) doğrusal bir denklem sistemi çıkar. Her bir $i$ nokta çifti için yazılan denklemler üst üste istiflenerek $2N \times 9$ boyutlarında bir $A$ matrisi oluşturulur:

$$A \cdot h = 0$$

Burada $h = [h_{11}, h_{12}, h_{13}, h_{21}, h_{22}, h_{23}, h_{31}, h_{32}, h_{33}]^T$ bilinmeyen parametreler vektörüdür. Bu sistemin önemsiz (*trivial*) $h=0$ çözümüne ulaşmasını engellemek amacıyla $\|h\|^2 = 1$ kısıtı getirilir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-03.png" alt="A Matrisi İstifleme ve Kısıtlı Denklem Yapısı" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 3: N adet nokta çiftinden oluşturulan 2N x 9 boyutundaki A matrisi ve ||h||² = 1 kısıtlı en küçük kareler denklemi.</em></figcaption>
  </div>
</figure>

Amacımız, $\|h\|^2 = 1$ kısıtı altında $\|A \cdot h\|^2$ ifadesini minimize eden $h$ vektörünü bulmaktır:

$$\min_{h} h^T A^T A h \quad \text{öyle ki} \quad h^T h = 1$$

Bu optimizasyonu çözmek için bir $\lambda$ Lagrange çarpanı eklenerek hata (*Loss*) fonksiyonu tanımlanır:

$$\mathcal{L}(h, \lambda) = h^T A^T A h - \lambda (h^T h - 1)$$

Hata fonksiyonunun $h$ vektörüne göre türevi alınıp sıfıra eşitlendiğinde karşımıza klasik Özdeğer/Özvektör (*Eigenvalue/Eigenvector*) problemi çıkar:

$$A^T A h = \lambda h$$

> **Nihai Çözüm:** Sistemi minimize eden $h$ vektörü, $A^T A$ matrisinin **en küçük özdeğerine (*smallest eigenvalue*) karşılık gelen özvektörüdür (*eigenvector*)**. Tekil Değer Ayrışımı (SVD) yöntemiyle $A = U \Sigma V^T$ ayrıştırıldığında $h$ vektörü, $V$ matrisinin son sütununa eşittir. Bu 9 elemanlı vektör 3x3 boyutuna getirilerek $H$ homografi matrisi elde edilir.

---

## 2. Aykırı Değerlerle Mücadele: RANSAC (Dealing with Outliers: RANSAC)

### 2.1 Aykırı Değer (Outlier) Problemi

SIFT gibi ilgi noktası dedektörleri, iki görüntüyü eşleştirirken sadece piksellerin yerel görsel görünümlerine (*descriptors*) bakar. Ancak tekrarlayan dokular, gölgeler veya gürültü nedeniyle, 3D uzayda aynı noktaya ait olmayan ancak görsel olarak birbirine çok benzeyen sahte eşleşmeler (*outliers*) kaçınılmaz olarak sisteme sızar.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-04.png" alt="Geçerli ve Hatalı Eşleşmeler" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 4: İki görüntü arasındaki doğru eşleşmeler (Inliers - Yeşil çizgiler) ve hatalı sahte eşleşmeler (Outliers - Kırmızı çizgiler).</em></figcaption>
  </div>
</figure>

```
  [Inliers (Geçerli Eşleşmeler)]           [Outliers (Aykırı/Hatalı Değerler)]
     Kameranın baktığı ortak                 Farklı nesnelerde yer alan ama
     3D sahne noktaları                      görsel olarak benzer sahte pikseller
```

Bu hatalı eşleşen noktalar doğrudan en küçük kareler denklemine dahil edilirse, tüm sistem geometrik olarak tamamen kayar. Bu nedenle, hesaplamaya başlamadan önce geçerli eşleşmeleri (*inliers*) hatalılardan (*outliers*) ayırmak şarttır.

### 2.2 RANSAC (RANdom SAmple Consensus) Algoritması

RANSAC, veri kümesindeki hatalı eşleşme (*outlier*) oranı %50'den fazla olsa dahi doğru modeli bulabilen son derece güçlü ve akıllı bir oylama algoritmasıdır.

Algoritmanın homografi hesaplama üzerindeki adımları şu şekildedir:

1. Veri kümesinden homografiyi çözmek için gereken minimum sayıda rastgele örnek seçilir ($s = 4$ nokta çifti).
2. Bu seçilen 4 nokta kullanılarak geçici bir $H$ homografi matrisi hesaplanır.
3. Tüm veri kümesindeki noktalar bu geçici $H$ matrisi ile hedef görüntüye yansıtılır. Yansıtılan konum ile gerçek koordinat arasındaki mesafe (hata pikseli) ölçülür. Hata payı belirlenen bir $\epsilon$ eşik değerinin altında kalan noktalar **Inlier** (geçerli değer) olarak kabul edilir ve oylama skoru ($M$) belirlenir.
4. Bu adımlar belirlenen bir $N$ iterasyon sayısı kadar tekrarlanır.
5. Süreç sonunda, en yüksek $M$ (*inlier*) oyunu alan homografi matrisi kazanan model olarak seçilir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-05.png" alt="En Küçük Kareler vs RANSAC 1. İterasyon" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 5: Klasik En Küçük Kareler fitting (Outlier'lar yüzünden kayar, Inlier: 2) vs. RANSAC 1. İterasyon (Inlier: 4).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-06.png" alt="RANSAC Kazanan İterasyon" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 6: RANSAC İterasyon i - Doğru model yakalandığında en yüksek inlier sayısına (Inlier: 20) ulaşılır.</em></figcaption>
  </div>
</figure>

> **Model İyileştirme (Refinement):** RANSAC kazanan modeli belirledikten sonra, sadece ilk seçilen 4 rastgele nokta ile yetinmek yerine, kazanan modelin belirlediği tüm $M$ adet inlier noktası bir araya getirilir ve Kısıtlı En Küçük Kareler yöntemiyle homografi matrisi en baştan çok daha hassas ve gürültüye dayanıklı bir şekilde yeniden hesaplanarak nihai hale getirilir.

---

## 3. Görüntü Yamultma ve Harmanlama (Warping and Blending)

Doğru homografi matrisi hesaplandıktan sonra, görüntüleri birleştirip kusursuz bir panorama haline getirmek için geometrik yamultma (*Warping*) ve fotometrik harmanlama (*Blending*) adımları uygulanır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-07.png" alt="Görüntü Yamultma Temel Konsepti" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 7: Görüntü Yamultma (Image Warping): Koordinat operatörü T(x,y) ile girdi görüntüsü f(x,y)'nin g(x,y) düzlemine bükülmesi.</em></figcaption>
  </div>
</figure>

### 3.1 İleri Doğru Yamultma (Forward Warping) ve Delik Problemi

Yamultma işleminde girdi görüntüsündeki her bir pikselin koordinatına $H$ dönüşümü uygulanır, hedef koordinat hesaplanır ve pikselin renk/parlaklık değeri hedefteki konuma yazılır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-08.png" alt="İleri Yamultma ve Piksel Izgarasında Delikler" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 8: İleri Doğru Yamultma (Forward Warping): Pikseller tam sayı olmayan konumlara düşer ve çıktıda boşluklar (holes) oluşur.</em></figcaption>
  </div>
</figure>

Ancak ileri doğru yamultmanın iki büyük kusuru vardır:
1. **Piksel Merkezine Oturmama:** Dönüştürülen koordinat genellikle hedef görüntüdeki tam sayı (*integer*) piksel merkezlerine denk gelmez.
2. **Delikler (Holes) ve Boşluklar:** Geometrik genişlemelerden dolayı, hedef görüntüdeki bazı pikseller hiçbir girdi pikseli tarafından hedef alınmaz. Çıktı görüntüsünde doldurulamamış siyah noktalar (delikler) oluşur.

### 3.2 Çözüm: Geriye Doğru Yamultma (Backward Warping)

Delik problemini kesin olarak çözmek için geriye doğru yamultma yöntemi uygulanır:

1. Girdi görüntüsünün 4 köşesine ileri doğru yamultma uygulanarak çıktı görüntüsünün sınır kutusu (*bounding box*) hesaplanır.
2. Sınır kutusu içindeki her bir çıktı pikseli $(x_d, y_d)$ tek tek taranır.
3. Her bir çıktı pikseli için TERS dönüşüm ($H^{-1}$) uygulanarak girdi görüntüsünde denk geldiği koordinat $(x_s, y_s)$ bulunur.
4. Bulunan koordinat tam sayı değilse, etraftaki piksellerden *Nearest Neighbor* veya *Bilinear Interpolation* ile renk değeri çekilir ve yazılır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-09.png" alt="Geriye Doğru Yamultma Şeması" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 9: Geriye Doğru Yamultma (Backward Warping): Hedef pikselden T⁻¹ ile girdi görüntüsüne dönüp interpolasyonla değer çekme.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-10.png" alt="Çoklu Görüntü Sınır Kutusu Hesaplama" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 10: Görsellerin köşe noktalarının referans tuvale bükülerek ortak sınır kutusunun (Bounding Box) belirlenmesi.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-11.png" alt="Ters Homografi ile Referans Tuvalden Görüntülere Erişim" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 11: Ters Homografiler (H₁₂, H₃₂) kullanılarak tuvaldeki pikseller için orijinal fotoğraflardan kesintisiz veri çekilmesi.</em></figcaption>
  </div>
</figure>

```
  [İleri Yamultma]   (x, y)   ──► H   ──► (x', y')   (Boşluklar ve delikler kalır)
  [Geri Yamultma]    (x', y') ──► H^-1 ──► (x, y)     (Kesintisiz, deliksiz çıktı)
```

Bu yöntemde çıktı görüntüsündeki her piksel geriye doğru taranarak doldurulduğu için çıktıda kesinlikle hiçbir delik veya boşluk oluşamaz.

### 3.3 Görüntü Harmanlama (Blending) ve Dikiş İzi (Seam) Problemi

Görüntüler geometrik olarak mükemmel hizalansa dahi, doğrudan üst üste bindirildiklerinde aralarında çok net keskin dikiş izleri (*hard seams*) görünür.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-12.png" alt="Doğrudan Bindirmede Keskin Dikiş İzi Oluşumu" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 12: İki görüntünün doğrudan üst üste konması (Hard overlay / Adım fonksiyonu ağırlıkları w₁, w₂) sonucu oluşan keskin dikiş izi.</em></figcaption>
  </div>
</figure>

Bu dikiş izlerinin iki temel fiziksel/optik nedeni vardır:
1. **Pozlama ve Işık Farklılıkları:** Görüntüler çekilirken kameranın otomatik pozlama (*exposure*) ayarlarının değişmesi veya sahnedeki anlık ışık değişimleri.
2. **Vinyet Etkisi (Vignetting):** Merceklerin fiziksel yapısı gereği, görüntünün merkezindeki piksellerin kenarlardaki piksellere kıyasla daha parlak olması (ışığın kenarlara doğru düşmesi).

İnsan görsel sistemi, özellikle düz çizgiler ve pürüzsüz konturlar üzerindeki 1 gri seviyelik çok küçük parlaklık değişimlerine karşı bile aşırı duyarlı olduğundan, bu dikiş izlerini anında fark eder. Basitçe örtüşen piksellerin ortalamasını almak (*averaging*) bu geçiş sınırlarını yumuşatsa da dikiş izlerini tamamen yok edemez.

### 3.4 Ağırlıklı Harmanlama (Weighted Blending)

Dikiş izlerini tamamen ortadan kaldırmak için piksellerin görüntünün merkezine olan yakınlığına göre ağırlıklandırıldığı bir geçiş fonksiyonu tanımlanır. İki görüntünün harmanlanmış piksel değeri ($I_{\text{blend}}$), yumuşak geçişli $w_1$ ve $w_2$ ağırlık matrisleri kullanılarak hesaplanır:

$$I_{\text{blend}} = \frac{w_1 I_1 + w_2 I_2}{w_1 + w_2}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-13.png" alt="Lineer Ağırlıklı Harmanlama Şeması" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 13: Yumuşak eğimli ağırlık fonksiyonları (w₁, w₂) ile ağırlıklı harmanlama (Weighted Blending) denklemi.</em></figcaption>
  </div>
</figure>

### 3.5 Mesafe Dönüşümü (Distance Transform) Tabanlı Harmanlama

Görüntü birleştirmede en başarılı ağırlık matrisleri Mesafe Dönüşümü (*Distance Transform* - örn. MATLAB `bwdist`) kullanılarak üretilir:

1. Bir pikselin ağırlığı, o pikselin görüntünün en yakın kenar sınırına olan fiziksel mesafesiyle doğru orantılı olarak atanır.
2. Piksel görüntünün ne kadar içindeyse (merkeze ne kadar yakınsa) optik kalitesi ve güvenilirliği o kadar yüksek kabul edilir ve harmanlamadaki ağırlığı ($w$) artar. Sınıra yakın piksellerin ağırlığı ise sıfıra doğru sönümlenir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-14.png" alt="Mesafe Dönüşümü Ağırlık Haritaları" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 14: Görüntü 1, 2 ve 3 için Mesafe Dönüşümü (Distance Transform) ile üretilen alfa ağırlık haritaları (w₁, w₂, w₃).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-15.png" alt="Ham Bindirme vs Harmanlanmış Panorama" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 15: (Üst) Pozlama izlerinin görüldüğü ham bindirme vs. (Alt) Distance Transform harmanlaması ile kusursuz kesintisiz panorama.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/homography-ransac-warping-and-blending-16.png" alt="Çoklu Fotoğraf Panoramik Mozaik Hizalaması" style="display:flex; border-radius: 5px; justify-content: center; width: 650px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 16: 6 adet kaynak görüntünün ikili homografiler ve geriye doğru yamultma / harmanlama ile tamamlanan panoramik mozaik birleşimi.</em></figcaption>
  </div>
</figure>

Bu akıllı ağırlıklandırma sayesinde, görüntüler arasındaki parlaklık geçişleri geniş alanlara yayılarak tamamen pürüzsüzleştirilir ve insan gözü tarafından hiçbir dikiş izi algılanamayan, tek parça halinde kusursuz bir geniş açılı panorama elde edilir.
