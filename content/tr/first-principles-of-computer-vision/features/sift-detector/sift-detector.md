# SIFT Tespiti ve Tanımlayıcı (SIFT Detector and Descriptor)

<!-- toc -->

## 1. Genel Bakış (Overview)

Geleneksel bilgisayarlı görü yaklaşımlarında, nesneleri tanımak ve konumlandırmak için **ikili bölütleme (binary segmentation)** ve **geometrik momentlerin analizi** oldukça etkilidir. Ancak bu yöntemler sadece son derece kontrol edilebilir endüstriyel ortamlarda (arkadan aydınlatmalı silüetler) veya yüksek kontrastlı metin okuma uygulamalarında (plaka tanıma vb.) kararlılık gösterir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-01.png" alt="Basit Şablon vs Karmaşık 2B Görünüm Eşleştirme" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 1: (Sol) Tekil ve izole şablon kapağı. (Sağ) Karmaşık, üst üste binmiş ve dönmüş CD kapaklarından oluşan gerçek dünya 2B sahnesi.</em></figcaption>
  </div>
</figure>

Gerçek dünya sahnelerinde yer alan üç boyutlu veya karmaşık iki boyutlu (planar) nesnelerin tanınması söz konusu olduğunda, bu basit yaklaşımlar tamamen çöker.

```
Geleneksel Şablon Eşleştirme (Template Matching) Sınırları:
──────────────────────────────────────────────────────────
1. Ölçek (Scale) Değişimi: Nesne derinliğine bağlı boyut değişimi.
2. Rotasyon (Rotation): Nesnenin 2D/3D dönme hareketleri.
3. Kısmi Tıkanma (Occlusion): Nesnenin bir kısmının engellenmesi.
4. Işık Değişimi (Illumination): Kamera kazancı, parlama ve gölgeler.
```

Eğer bir nesneyi aratmak için klasik **şablon eşleştirme (template matching)** veya **normalize çapraz korelasyon (normalized cross-correlation - NCC)** kullanılmak istenirse, nesnenin olası tüm dönme açıları ve farklı ölçek varyasyonları için binlerce alt-şablon (partial templates) üretilip tüm görüntü üzerinde kaydırılarak aranması gerekir. Bu süreç, hesaplama karmaşıklığı açısından $O(N \cdot M \cdot S \cdot R)$ seviyesine ulaşır ve pratik uygulamalar için tamamen imkansızdır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-02.png" alt="Rotasyon ve Aydınlatma Değişimi Altında Görünüm" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 2: Aynı nesnenin düz duruşu (sol) ile döndürülmüş ve ışık açısı değişmiş duruşu (sağ). Yerel penceredeki piksel değerleri doğrudan eşleştirilemez.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-03.png" alt="Yakınlaştırılmış Piksel Yamalarının Karşılaştırılması" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 3: Yakınlaştırılmış lokal piksel yaması. Nesne döndüğünde piksellerin matris dizilimi tamamen değiştiği için doğrudan piksel farkı almak başarısız olur.</em></figcaption>
  </div>
</figure>

> **Temel Sezgi:** Bu temel problemin aşılması, görüntüden doğrudan son derece ayırt edici, benzersiz ve geometrik/aydınlatma değişimlerine karşı dayanıklı **yerel öznitelikler (highly descriptive and unique local features)** çıkarılmasına bağlıdır. Bu özniteliklerin konumları ve yerel görünüm imzaları (descriptors) çıkarıldıktan sonra, iki farklı görüntüdeki noktalar birebir eşleştirilerek nesne tanıma, panorama birleştirme (image stitching) ve 3D rekonstrüksiyon işlemleri başarıyla gerçekleştirilir.

---

## 2. İlgi Noktası Nedir? (What is an Interest Point?)

Bir görüntünün **ilgi noktası (interest point)**, yerel olarak en zengin görsel bilgiye ve benzersizliğe sahip olan bölgesidir. Yerel bir yamanın ilgi noktası olarak seçilebilmesi için belirli kritik kriterleri karşılaması gerekir:

### İdeal Bir İlgi Noktasının Nitelikleri:
- **Zengin İçerik (Rich Content):** Yerel analiz penceresi içinde parlaklık (renk/yoğunluk) varyasyonunun yüksek olması gerekir.
- **Net Temsil Edilebilirlik (Well-defined Representation):** Noktanın etrafındaki görsel dokudan, eşleştirmede kullanılacak benzersiz ve kompakt bir imza (descriptor) üretilebilmelidir.
- **Kesin Konumlandırma (Well-defined Position):** Eşleştirmenin uzamsal doğruluğu için ilgi noktasının görüntü düzleminde net bir koordinatı ($x, y$) bulunmalıdır.
- **Ölçek ve Rotasyon Değişmezliği (Scale & Rotation Invariance):** Nesne büyüdüğünde, küçüldüğünde veya döndüğünde bile aynı koordinat ve imza kararlı bir şekilde tekrar üretilebilmelidir (repeatability).
- **Işığa Karşı Dayanıklılık (Insensitivity to Illumination):** Gölgelerden, parlamalardan ve kamera kazancından etkilenmemelidir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-04.png" alt="Homojen ve Düz Doku Yamaları" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 4: Düz ve homojen dokulu yamalar (ahşap dokusu/düz yüzey). İçerisinde gradyan varyasyonu olmadığı için ilgi noktası olamazlar.</em></figcaption>
  </div>
</figure>

### Çizgi, Kenar, Köşe ve Lekelerin Karşılaştırılması:

1. **Kenarlar (Edges):** Kenarlar, yoğunluğun tek bir doğrultuda hızlı değiştiği bölgelerdir. Bir kenar çizgisi boyunca yerel analiz penceresi kaydırıldığında görünüm neredeyse hiç değişmez (**aperture problem / açıklık problemi**). Bu belirsizlik nedeniyle kenarlar iyi birer ilgi noktası değildir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-05.png" alt="Kenar Tespiti ve Açıklık Problemi" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 5: Kenar boyunca kaydırma belirsizliği (Aperture Problem). Pencere kenar çizgisi üzerinde hareket ettirildiğinde pikseller değişmez, kesin uzamsal konum tespit edilemez.</em></figcaption>
  </div>
</figure>

2. **Köşeler (Corners):** Köşeler iki farklı yöndeki kenarın birleşimi olduğu için uzamsal konumları net olarak saptanabilir ($x, y$). Ancak, karmaşık dokular barındıran nesneleri temsil edecek kadar zengin yerel görünüm (appearance) bilgisi sunamazlar ve görüntüde seyrek bulunurlar.
3. **Lekeler ve Yamalar (Blobs):** Belirli bir uzamsal ölçeğe ($\sigma$), baskın bir yöne ($\theta$) ve zengin yerel doku varyasyonuna sahip olan dairesel/oval yamalardır. Konumları, boyutları ve iç dokuları matematiksel olarak kararlı modellenebildiği için bilgisayarlı görüde en ideal ilgi noktası adayı **Blob** yapılarıdır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-06.png" alt="Köşe ve Leke Yamalarının İncelemesi" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 6: Köşe ve leke yamalarının karşılaştırılması. Leke yamaları hem uzamsal konumu hem de ölçek penceresini net olarak tanımlar.</em></figcaption>
  </div>
</figure>

---

## 3. Leke Tespiti (Detecting Blobs)

Matematiksel olarak bir blobu saptamak, farklı uzamsal çözünürlüklerde (**scale-space / ölçek uzayı**) yerel parlaklık ekstremumlarını (peaks/tepe noktaları) bulmak demektir.

### 3.1 1 Boyutlu Sinyalde İkinci Türev ve Ölçek Uzayı (Scale Space)

Tek boyutlu bir sinyalde gürültüyü süzmek için $\sigma$ genişliğindeki Gauss filtresi kullanılır:

$$G(x, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-07.png" alt="1D Sinyal ve Gauss Yumuşatma" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 7: (Üstten alta) Gürültülü adım sinyali $f$, Gauss yumuşatma çekirdeği $n_\sigma$ ve yumuşatılmış sinyal $n_\sigma * f$.</em></figcaption>
  </div>
</figure>

Sinyal, Gauss'un birinci türeviyle ($\frac{d}{dx} G_\sigma$) konvolüsyona sokulduğunda kenar geçişlerinde birer tepe noktası (peak) verir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-08.png" alt="Gauss Birinci Türevi ile Kenar Yanıtı" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 8: Gauss'un 1. türevi $\nabla(n_\sigma)$ filtre yanıtı. Kenarın tam üzerinde maksimum genlik (peak) oluşturur.</em></figcaption>
  </div>
</figure>

Gauss'un ikinci türevi filtresi ($\frac{d^2}{dx^2} G_\sigma$ / Inverted Mexican Hat) uygulandığında ise kenarın tam merkezinde bir **Sıfır Geçişi (Zero-Crossing)** oluşur.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-09.png" alt="Gauss İkinci Türevi ve Sıfır Geçişi" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 9: Gauss'un 2. türevi $\nabla^2(n_\sigma)$ filtresi ve sinyalle konvolüsyonu. Kenar merkezinde tam sıfır geçişi (zero-crossing) gözlenir.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-10.png" alt="1D Blob Yapı Örnekleri" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 10: 1D sinyaldeki farklı blob benzeri (pulse, bump, trough) temel yapılar.</em></figcaption>
  </div>
</figure>

Farklı genişliklerdeki blobları (örneğin genişliği sırasıyla $W$, $2W$ ve $3W$ olan $A, B, C$ blobları) analiz etmek için filtre genişliğini ($\sigma$) sürekli değiştirerek görüntü çözünürlüğünü düşürdüğümüz bir **Ölçek Uzayı (Scale Space)** tasarlanır:

$$S(x, \sigma) = f(x) * G(x, \sigma)$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-11.png" alt="Farklı Genişlikteki Bloblar Üzerinde Filtre Yanıtları" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 11: Farklı genişlikteki Bloblar ($A, B, C$) üzerinde Gauss yumuşatma, 2. türev ve normalleştirilmemiş yanıtlar. Normalleştirme yapılmazsa geniş ölçeklerde yanıt genliği düşer.</em></figcaption>
  </div>
</figure>

### 3.2 $\sigma^2$-Normalizasyonu ve Karakteristik Ölçek (Characteristic Scale)

Gauss filtresinin standart sapması ($\sigma$) büyüdükçe (ölçek arttıkça), filtrenin tepe genlik değeri düşer ve dolayısıyla sinyal yanıtı sönümlenir. Farklı ölçeklerde elde edilen ekstremum yanıt genliklerini birbiriyle tutarlı şekilde karşılaştırabilmek için ikinci türev filtresi $\sigma^2$ sabiti ile çarpılarak normalleştirilir. Buna **$\sigma$-normalleştirilmiş çıktı** denir:

$$\text{NLoG}_{1D} = \sigma^2 \frac{d^2 G_\sigma}{dx^2} * f(x)$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-12.png" alt="Karakteristik Ölçek ve Yerel Ekstremumlar" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 12: $\sigma^2$-normalleştirilmiş NLoG yanıtının blobların tam merkezinde en yüksek ekstremumu (tepe noktasını) oluşturması.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-13.png" alt="Blob Boyutu ile Karakteristik Ölçek İlişkisi" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 13: Karakteristik Ölçek ($\sigma^*$): $A$ bloğu için $\sigma_1$, $B$ bloğu için $2\sigma_1$, $C$ bloğu için $3\sigma_1$ seviyesinde maksimum yanıt alınır.</em></figcaption>
  </div>
</figure>

Eğer farklı $\sigma$ (ölçek) değerlerine bağlı olarak bir blobun tam merkezindeki yanıt genliği grafiğe dökülürse, yanıtın tam olarak $\sigma^* \propto \text{Blob Genişliği}$ oranında maksimum (yerel ekstremum) yaptığı görülür:

- **$A$ bloğu ($Genel=W$):** $\sigma_A^* = \sigma_1$ ölçeğinde ekstremum verir.
- **$B$ bloğu ($Genel=2W$):** $\sigma_B^* = 2\sigma_1$ ölçeğinde ekstremum verir.
- **$C$ bloğu ($Genel=3W$):** $\sigma_C^* = 3\sigma_1$ ölçeğinde ekstremum verir.

> **Karakteristik Ölçek (Characteristic Scale):** Bu maksimum yanıtın elde edildiği benzersiz $\sigma^*$ değerine o blobun Karakteristik Ölçeği denir. Böylece 2 boyutlu $(x, \sigma)$ uzayında yerel ekstremumları arayarak hem blobun kesin konumunu ($x^*$) hem de blobun boyutunu ($\sigma^*$) saptamış oluruz.

### 3.3 2 Boyutlu Uzayda NLoG Operatörü

İki boyutlu görüntülerde tek boyuttaki normalleştirilmiş ikinci türevin karşılığı **Normalleştirilmiş Laplacian of Gaussian (NLoG)** operatörüdür. 2D Gauss fonksiyonuna Laplacian operatörünün ($\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$) uygulanması ve $\sigma^2$ normalizasyonu ile elde edilir:

$$\text{NLoG}_{2D} = \sigma^2 \nabla^2 G(x, y, \sigma) = \sigma^2 \left( \frac{\partial^2 G}{\partial x^2} + \frac{\partial^2 G}{\partial y^2} \right)$$

$$\text{NLoG}_{2D}(x, y, \sigma) = -\frac{1}{2\pi\sigma^2} \left( 2 - \frac{x^2 + y^2}{\sigma^2} \right) e^{-\frac{x^2+y^2}{2\sigma^2}}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-14.png" alt="2D Filtre Operatörleri: Laplacian, Gaussian, LoG, NLoG" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 14: 2B Filtre Operatörlerinin 3B yüzey görünümleri: Laplacian ($\nabla^2$), Gaussian ($n_\sigma$), LoG ($\nabla^2 n_\sigma$) ve Normalleştirilmiş NLoG ($\sigma^2 \nabla^2 n_\sigma$).</em></figcaption>
  </div>
</figure>

2D bir görüntüdeki tüm blobları saptamak için görüntü bu NLoG filtresiyle çok sayıda farklı ölçekte ($\sigma$) konvolüsyona sokularak 3 boyutlu bir **Ölçek-Uzay Hacmi (Scale-Space Volume)** oluşturulur:

$$V(x, y, \sigma) = I(x, y) * \left[ \sigma^2 \nabla^2 G(x, y, \sigma) \right]$$

Bu 3B hacim içinde yerel ekstremum ($x^*, y^*, \sigma^*$) noktaları aranır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-15.png" alt="Ölçek Uzayı Görselleştirmesi" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 15: Düşen adam resmi üzerinde Ölçek Uzayı (Scale-Space) serisi: $S(x,y,\sigma_0) \dots S(x,y,\sigma_3)$. $\sigma$ büyüdükçe detaylar kaybolur ve çözünürlük düşer.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-16.png" alt="Zengin Dokulu Bölgede Karakteristik Ölçek Ekstremumu" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 16: Düşen adamın göz bölgesinde ölçek boyunca NLoG yanıtı. $\sigma_1$ ölçeğinde belirgin bir ekstremum tepe noktası oluşur (Lindeberg 1994).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-17.png" alt="Homojen Bölgede Ekstremum Oluşmaması" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 17: Düz/homojen (pantolon paçası yanındaki arka plan) bir noktada ölçek boyunca NLoG yanıtı. Güçlü bir ekstremum oluşmadığı için leke olarak kabul edilmez.</em></figcaption>
  </div>
</figure>

---

## 4. SIFT Dedektörü (SIFT Detector)

David Lowe tarafından önerilen **SIFT (Scale-Invariant Feature Transform)** dedektörü, yukarıdaki teorik NLoG tabanlı blob tespitini donanımsal olarak son derece hızlı, verimli ve gürültüye dayanıklı hale getiren bir dizi mühendislik yaklaşımı (tricks) içerir.

### 4.1 Hızlı NLoG Yaklaşımı: Difference of Gaussians (DoG)

Her ölçek seviyesinde 2D NLoG filtresini sıfırdan hesaplayıp görüntüyle konvolüsyona sokmak çok yüksek işlem gücü gerektirir. Lowe, ölçek uzayındaki iki ardışık Gauss pürüzsüzleştirilmiş görüntüsünün birbirinden çıkarılmasıyla elde edilen **Difference of Gaussians (DoG - Gaussların Farkı)** operatörünün, NLoG operatörüne mükemmel bir matematiksel yaklaşım sunduğunu ispatlamıştır:

$$\text{DoG}(x, y, \sigma) = S(x, y, k\sigma) - S(x, y, \sigma) = I(x, y) * \left[ G(x, y, k\sigma) - G(x, y, \sigma) \right]$$

Isı yayılımı ve difüzyon denklemlerinden yararlanarak limit durumunda şu yaklaşım elde edilir:

$$\frac{\partial G}{\partial \sigma} = \lim_{\Delta\sigma \to 0} \frac{G(x,y,\sigma + \Delta\sigma) - G(x,y,\sigma)}{\Delta\sigma}$$

$$\sigma \nabla^2 G = \frac{\partial G}{\partial \sigma} \approx \frac{G(x,y,k\sigma) - G(x,y,\sigma)}{(k-1)\sigma}$$

Buradan her iki tarafı $\sigma$ ile çarparak $\sigma$-normalleştirilmiş Laplacian elde edilir:

$$G(x,y,k\sigma) - G(x,y,\sigma) \approx (k-1) \cdot \left[ \sigma^2 \nabla^2 G \right] = (k-1) \cdot \text{NLoG}$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-18.png" alt="NLoG ve DoG Eğrilerinin Karşılaştırılması" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 18: Tam normalleştirilmiş NLoG eğrisi ile DoG yaklaşımının birebir çakışması ($DoG \approx (s-1)\text{NLoG}$).</em></figcaption>
  </div>
</figure>

Bu matematiksel ilişki sayesinde, sadece Gauss pürüzsüzleştirilmiş görüntülerin birbirinden çıkarılmasıyla, çok daha ağır bir işlem olan NLoG çıktısı (sadece sabit bir $k-1$ ölçek faktörü farkıyla) son derece hızlıca elde edilir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-19.png" alt="DoG Piramidinin İnşası" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 19: Görüntü $I(x,y)$ Gauss ölçek uzayından geçirilir ve ardışık seviyelerin birbirinden çıkarılmasıyla DoG fark görüntüleri yığını elde edilir (Lowe 2004).</em></figcaption>
  </div>
</figure>

### 4.2 3 Boyutlu Ekstremum Arama ve Zayıf Noktaların Elenmesi

DoG fark görüntüleri yığını (stack) oluşturulduktan sonra yerel ekstremumları saptamak için şu adımlar izlenir:

1. DoG hacmi üzerinde $3 \times 3 \times 3$ boyutlarında küçük kübik bir pencere kaydırılır.
2. Merkezdeki pikselin mutlak değeri, kendi görüntüsündeki 8 komşusuyla ve bir üst ile bir alt ölçek seviyesindeki 9'ar komşusuyla (toplam **26 komşuyla**) karşılaştırılır.
3. Eğer merkez piksel bu 26 komşunun tamamından kesin olarak büyük veya küçükse (yerel ekstremum ise) bir ilgi noktası adayı olarak işaretlenir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-20.png" alt="3B Komşulukta Ekstremum Arama" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 20: DoG hacminde $3 \times 3 \times 3$ komşuluğundaki 26 piksel ile merkez pikselin karşılaştırılması.</em></figcaption>
  </div>
</figure>

**Zayıf Noktaların Temizlenmesi:** Gürültü ve düşük kontrast içeren kararsız adayları süzmek için belirlenen bir eşik değerinin altındaki ekstremum pikselleri elenerek supprese edilir. Ayrıca kenar üzerindeki kararsız noktalar Hessian matrisinin özdeğer oranları kullanılarak temizlenir. Geriye kalan güçlü ve kararlı pikseller kesin SIFT İlgi Noktaları olarak saptanır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-21.png" alt="Kararlı SIFT Noktalarının Seçimi" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 21: Zayıf ve kararsız ekstremumların elenmesiyle görüntü üzerinde kararlı SIFT dairelerinin (konum ve ölçek yarıçapı) elde edilmesi (Lowe 2004).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-22.png" alt="God of War Kapak Resminde SIFT Noktaları" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 22: PS2 God of War kapak resmi üzerinde farklı ölçek yarıçaplarında ($r \propto \sigma^*$) tespit edilmiş SIFT ilgi halkaları.</em></figcaption>
  </div>
</figure>

### 4.3 Ölçek ve Rotasyon Değişmezliğinin Sağlanması

#### 1. Ölçek Değişmezliği (Scale Invariance)
Nesnenin kameraya uzaklığına bağlı olarak değişen büyütme oranları (magnifications), DoG spektrumunda farklı $\sigma^*$ karakteristik ölçeklerinde tepe değerleri üretilmesine neden olur. Bu karakteristik ölçeklerin oranı nesneler arasındaki gerçek boyut oranını ($\frac{\sigma_1^*}{\sigma_2^*}$) verir. SIFT, eşleştirmeden önce ilgi noktası pencerelerini bu karakteristik ölçeklerine göre yeniden boyutlandırarak (normalize ederek) ölçek farkını tamamen ortadan kaldırır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-23.png" alt="Ölçek Oranının Karakteristik Ölçeklerle Tespiti" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 23: Farklı mesafelerden çekilmiş aynı nesne için Karakteristik Ölçeklerin Oranı ($\frac{\sigma_1^*}{\sigma_2^*}$) doğrudan ölçek değişim oranını verir (Mikolajczyk 2001).</em></figcaption>
  </div>
</figure>

#### 2. Rotasyon Değişmezliği ve Birincil Yönelim (Principal Orientation)
Ölçek normalizasyonu yapılmış dairesel ilgi noktası bölgesini kapsayan kare bir piksel penceresi tanımlanır.

1. Pencere içindeki her bir piksel için yatay ($I_x$) ve dikey ($I_y$) kısmi türevler üzerinden yerel gradyan büyüklüğü ($m$) ve yön açısı ($\theta$) hesaplanır:

$$m(x,y) = \sqrt{I_x^2 + I_y^2} \quad \text{ve} \quad \theta(x,y) = \tan^{-1}\left( \frac{I_y}{I_x} \right)$$

2. Işık değişimlerine, gölgelere ve kamera kazancına karşı bağışıklık kazanmak için gradyan büyüklükleri tamamen ihmal edilir ve sadece gradyan yönleri ($\theta$) hesaba katılır.
3. Açısal aralık ($0^\circ - 360^\circ$) 36 dilime bölünerek bir **Gradyan Yönelim Histogramı** oluşturulur.
4. Bu histogramdaki en yüksek tepe noktası, o ilgi noktasının **Birincil Yönelimi (Principal Orientation)** olarak atanır.
5. Eşleştirme aşamasında, ilgi noktası etrafındaki yama (patch), bu birincil yönelim açısı kadar ters yönde döndürülerek (reoriented) her zaman **Kuzey (Yukarı)** yönüne hizalanır. Böylece rotasyon etkisi tamamen sıfırlanmış olur.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-24.png" alt="Birincil Yönelim Histogramı" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 24: (Sol) Yerel yamadaki piksellerin gradyan yön vektörleri. (Sağ) 36 dilimli yönelim histogramı ve tepe noktası seçimi.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-25.png" alt="Döndürülmüş Nesnede Birincil Yönelim Hizalaması" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 25: Döndürülmüş CD kapağında ana yönelim okunun tespit edilerek yamanın standart dik konuma döndürülmesi.</em></figcaption>
  </div>
</figure>

---

## 5. SIFT Tanımlayıcısı (SIFT Descriptor)

Boyut ve rotasyon etkileri tamamen sıfırlandıktan sonra, normalize edilmiş ve kuzeye hizalanmış ilgi noktası yamasının iç görünümünü temsil edecek kompakt ve güçlü bir yerel imza (signature) üretilmelidir.

### 5.1 SIFT Descriptor'ın Matematiksel İnşası

1. İlgi noktasının etrafındaki normalize edilmiş yama üzerinde piksellerden oluşan standart boyutta bir ızgara (grid) kurulur.
2. Yine ışık ve kontrast değişimlerine bağışıklık sağlamak adına gradyan büyüklükleri ihmal edilerek, her pikselin sadece gradyan yönleri ($\theta$) hesaplanır.
3. Bu yama alanı, örtüşmeyen 4 eşit çeyreğe (quadrant) bölünür.
4. Her bir çeyrek için bağımsız olarak, 8 ana yönü ($0^\circ, 45^\circ, 90^\circ, \dots, 315^\circ$) kapsayan 8-binli yerel gradyan yönelim histogramı hesaplanır.
5. Bu 4 ayrı histogram yan yana birleştirilerek (**concatenated**) tek bir uzun vektöre dönüştürülür.
6. Lowe'un orijinal patentli uygulamasında $16 \times 16$ piksel alanı, $4 \times 4 = 16$ alt bölgeye bölünür ve her alt bölge için 8 yönlü histogram hesaplanır. Bu sayede **$16 \times 8 = 128$ boyutlu meşhur SIFT Descriptor vektörü** üretilmiş olur.

```
 Izgara Yapısı (Grid)                  4 Çeyrek Histogramı
 ┌──────────┬──────────┐  
 │          │          │                Lokal Hist 1 ──┐
 │ Çeyrek 1 │ Çeyrek 2 │                Lokal Hist 2 ──┼──► Concatenate ──► [ SIFT Tanımlayıcı Vektörü ]
 │          │          │                Lokal Hist 3 ──┼──►   (Uzun Birleşik Histogram)
 ├──────────┼──────────┤                Lokal Hist 4 ──┘
 │          │          │
 │ Çeyrek 3 │ Çeyrek 4 │
 │          │          │
 └──────────┴──────────┘
```

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-26.png" alt="SIFT Descriptor Vektör İnşası" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 26: SIFT Tanımlayıcısının oluşumu: Hizalanmış pencere alt bölgelere ayrılır, her bölgenin yön histogramı hesaplanır ve birleştirilerek 128D imza oluşturulur.</em></figcaption>
  </div>
</figure>

### 5.2 İki SIFT Tanımlayıcısının Karşılaştırılmasında Kullanılan Metrikler ($H_1, H_2$)

1. **L2 Mesafesi (L2 Distance - Öklid):**
   İki histogram arasındaki farkların karelerinin toplamının kareköküdür. Mesafe sıfıra ne kadar yakınsa, yerel dokuların eşleşmesi o kadar kusursuzdur:

   $$D(H_1, H_2) = \sqrt{\sum_{k} \left( H_1[k] - H_2[k] \right)^2}$$

2. **Normalize Korelasyon (Normalized Correlation):**
   Tanımlayıcıların ortalamaları ($\mu_1, \mu_2$) çıkarılarak kendi enerjilerine bölünmesiyle hesaplanır. Değerin 1.0 çıkması mükemmel bir doğrusal uyumu gösterir:

   $$D(H_1, H_2) = \frac{\sum_{k} (H_1[k] - \mu_1)(H_2[k] - \mu_2)}{\sqrt{\sum_{k} (H_1[k] - \mu_1)^2 \sum_{k} (H_2[k] - \mu_2)^2}} \quad \text{burada} \quad \mu = \frac{1}{N} \sum_{k} H[k]$$

3. **Kesişim Metriği (Intersection Metric):**
   Histogramların her bir kutusu (bin) için minimum değerlerinin toplanmasıyla elde edilen örtüşme (overlap) miktarıdır:

   $$D(H_1, H_2) = \sum_{k} \min\left( H_1[k], H_2[k] \right)$$

### 5.3 SIFT Eşleştirme Örnekleri ve Uygulamalar

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-27.png" alt="Ölçek Değişiminde SIFT Eşleştirmesi" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 27: Büyük ölçek farkı içeren görüntülerde (Donnie Darko DVD ve God of War kapakları) birebir SIFT eşleşme hatları.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-28.png" alt="Rotasyon Altında SIFT Eşleştirmesi" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 28: $45^\circ$, $90^\circ$ ve ters dönmüş ($180^\circ$) Michel Gondry CD kapağında kararlı SIFT eşleşmeleri.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-29.png" alt="Karmaşık Yığın ve Kısmi Tıkanmada SIFT Eşleştirmesi" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 29: Üst üste binmiş karmaşık CD kapakları (clutter & occlusion) arasında aranılan nesnenin SIFT ile tespiti.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-30.png" alt="Dağ Fotoğraflarında SIFT Noktası Eşleştirme" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 30: İki dağ manzarası fotoğrafındaki ortak SIFT noktalarının otomatik eşleştirilmesi (Autostitch).</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-31.png" alt="Panorama Dikme ve Dönüştürme" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 31: Eşleşen SIFT noktaları kullanılarak fotoğrafların geometrik olarak dönüştürülmesi (warp) ve panorama oluşturulması.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-32.png" alt="30 Fotoğraftan Devasa Kolaj Oluşturma" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 32: Cam arkasından çekilmiş 30 farklı kareden SIFT eşleştirmesi ile birleştirilmiş devasa iç/dış mekan kolajı (Nomura 2007).</em></figcaption>
  </div>
</figure>

### 5.4 SIFT Algoritmasının Sınırları ve 3D Nesne Sıkıntısı

SIFT, iki boyutlu düzlemsel (planar) nesnelerde, farklı rotasyonlar, büyük ölçek değişimleri ve ağır tıkanmalar (occlusions) altında yüzlerce kararlı eşleşme üretebilir. Ortak noktalar üzerinden görüntüleri geometrik olarak warp ederek kesintisiz panoramalar ve kolajlar dikmeyi sağlar.

Ancak **üç boyutlu (3D) nesnelerin tanınmasında SIFT başarısız olmaya başlar**.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/sift-detector-33.png" alt="3B Bakış Açısı Değişiminde SIFT Sınırı" style="display:flex; border-radius: 5px; justify-content: center; width: 600px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 33: 3B nesnelerde bakış açısı (viewpoint) etkisi: Açı farkı $0^\circ$ (kusursuz eşleşme), $30^\circ$ (dramatik düşüş), $90^\circ$ (eşleşmenin tamamen çökmesi).</em></figcaption>
  </div>
</figure>

Bunun nedeni, 3D bir nesneye farklı bakış açılarından (viewpoints) bakıldığında, yerel özniteliklerin 3B derinlik geometrisi yüzünden tamamen değişmesidir. Deneysel sonuçlar göstermiştir ki:

- **Bakış açısı 30 derece değiştiğinde:** Eşleşen SIFT noktalarında dramatik bir düşüş yaşanır.
- **Bakış açısı farkı 90 dereceye ulaştığında:** Neredeyse hiç eşleşen SIFT noktası elde edilemez.

> **Sonuç:** SIFT, 3D nesnelerde sadece çok küçük bakış açısı değişimleri altında güvenilirdir.

---

## 6. Özetleyici Teknik Karşılaştırma Tablosu

| Konu Başlığı | Temel Matematiksel Denklem | Saptadığı / Tanımladığı Değer | Çözdüğü Kritik Görü Problemi | Karşılaştığı Temel Kısıt / Sınır |
| :--- | :--- | :--- | :--- | :--- |
| **İlgi Noktası** | Dairesel yama (Blobs) | Yerel konumsal koordinat ($x, y$), ölçek penceresi ($\sigma$) ve yönelim ($\theta$). | Kenarların çizgi boyunca kayma belirsizliğini ve köşelerin seyrekliğini giderir. | Görüntüde hiçbir dokunun olmadığı tamamen homojen (flat) alanlar. |
| **Blob Tespiti** | $\text{NLoG} = \sigma^2 \nabla^2 G$ | Karakteristik Ölçek ($\sigma^*$) ve konum ($x^*, y^*$). | Farklı boyutlardaki nesneleri ölçek uzayında yerel ekstremumla yakalar. | Yüksek boyutlu Gauss entegrallerinin piksel başına düşen işlem maliyeti. |
| **SIFT Dedektörü** | $\text{DoG} = S(k\sigma) - S(\sigma)$ | Ölçek ve rotasyondan arındırılmış ilgi noktaları. | Hızlı DoG yaklaşımıyla işlem yükünü sönümler ve birincil yön tayini yapar. | Keypoint adayları arasındaki gürültülü ve zayıf ekstremumlar. |
| **SIFT Tanımlayıcısı** | Vektör birleştirme (Concatenation) | 128 boyutlu benzersiz görsel imza vektörü. | Kısmi tıkanma, ışık değişimleri ve gürültü altında kararlı eşleştirme sağlar. | 3D nesnelerde 30° ve 90° bakış açısı değişimlerinde eşleşmenin tamamen çökmesi. |
