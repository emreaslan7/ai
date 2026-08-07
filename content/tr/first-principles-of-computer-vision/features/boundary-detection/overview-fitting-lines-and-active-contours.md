# Genel Bakış, Doğru ve Eğri Uydurma, Aktif Konturlar

<!-- toc -->

Bu teknik ders notu, bilgisayarlı görünün temel aşamalarından olan **Sınır Tespiti (Boundary Detection)** konusunu; gürültülü kenar piksellerinden sürekli nesne hatlarına geçiş zorlukları, **En Küçük Kareler Doğru ve Eğri Uydurma (Least Squares Line and Curve Fitting)** matris çözümleri, dikey çizgi tekillikleri ve dinamik birer esnek eğri olan **Aktif Konturlar (Active Contours / Snakes)** çerçevesinde matematiksel türetimleri ve pratik algoritmalarıyla detaylıca ele almaktadır.

---

## 1. Sınır Tespitine Genel Bakış (Overview)

Kenar tespiti (*edge detection*) aşamasında elde edilen çıktılar, genellikle kesikli piksellerden, parazit gürültülerden ve karmaşık arka plan çizgilerinden oluşur. Bilgisayarlı görünün temel amacı ise bu pikselleri birleştirerek nesnelerin sınırlarını (silüetlerini) sürekli birer geometrik çizgi veya kapalı eğri halinde ortaya çıkarmaktır. Bu probleme **Sınır Tespiti (Boundary Detection)** adı verilir.

<figure style="display:flex; justify- content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-01.png" alt="Boundary Detection Pipeline on Antique Vase" style="display:flex; border-radius: 5px; justify-content: center; width: 650px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 1: Antik vazo görüntüsü üzerinde kenar tespitinden eşiklemeye, morfolojik filtrelerden inceltmeye ve nihai sürekli sınır tespitine uzanan işlem hattı.</em></figcaption>
  </div>
</figure>

### 1.1. Kenar Tespiti ve Sınır Tespiti Arasındaki Fark

- **Kenar Tespiti (Edge Detection):** Görüntüdeki lokal parlaklık değişimlerini (gradyan büyüklüklerini) piksel seviyesinde saptayan yerel (*local*) bir işlemdir. Çıktı ikili kenar haritasıdır.
- **Sınır Tespiti (Boundary Detection):** İkili kenar piksellerini küresel (*global*) yapısal bir nesne hat veya parametrik eğri olarak birleştiren anlamsal ve geometrik bir süreçtir.

### 1.2. Karşılaşılan Başlıca Zorluklar

Sınır tespiti algoritmaları, gerçek dünya görüntülerinde şu üç temel fiziksel ve geometrik problemle baş etmek zorundadır:

1. **Dışsal (Alakasız) Kenarlar (Extraneous Data):** Görüntüde aranan nesnenin sınırları dışında, arka plandaki dokular, yüzey desenleri veya gölgeler yüzünden oluşmuş binlerce alakasız kenar pikseli bulunur. Algorithma hangilerinin hedef nesneye ait olduğunu ayırt etmelidir.
2. **Eksik/Yetersiz Veri ve Tıkanmalar (Incomplete Data / Occlusions):** Aydınlatma yetersizliği, nesnenin kendi düşük kontrastlı dokusu veya başka bir nesnenin arkasında kalması (*occlusion*) yüzünden sınır kenarlarının bir kısmı algılanamaz ve sınır hatlarında büyük boşluklar (*gaps*) oluşur.
3. **Görüntü Gürültüsü (Noise):** Sensör gürültüsü nedeniyle gerçekte sınır olmayan yerlerde sahte kenar pikselleri oluşurken, gerçek kenar koordinatları uzamsal olarak kayabilir.

---

## 2. Doğru ve Eğri Uydurma (Fitting Lines and Curves)

En temel sınır tespiti problemi, gürültülü ve ayrık kenar noktaları kümesine parametrik bir doğru veya düşük dereceli bir polinom eğrisi uydurmaktır (*curve fitting*).

### 2.1. Kenar Görüntülerinin Ön İşlemesi (Preprocessing Pipeline)

Ham bir görüntüden temiz sınırlara ulaşmak için doğrudan uydurma işlemine geçilmez. Süreç adım adım şu ön işlemlerden geçer:

1. **Kenar Tespiti ve Eşikleme (Edge Detection & Thresholding):** Görüntüye bir kenar operatörü (örneğin Sobel) uygulanarak her pikselde gradyan büyüklüğü hesaplanır ve bu harita eşiklenerek ikili (*binary*) bir kenar görüntüsü elde edilir.
2. **Büzme ve Genişletme (Shrink & Expand):** İkili morfolojik işlemlerden olan büzme (*shrinking*) uygulanarak izole kalmış küçük gürültü pikselleri yok edilir. Ardından kalan pikseller tekrar genişletilerek (*expanding*) kenar sürekliliği korunur.
3. **İnceltme (Thinning):** Kalınlaşan kenar hatları tek piksel genişliğine indirilerek doğru ve eğri uydurma için kararlı $(x_i, y_i)$ koordinat verileri hazırlanır.

```mermaid
flowchart LR
    A["Giriş Görüntüsü"] --> B["Kenar Tespiti & Eşikleme"]
    B --> C["Shrink & Expand (Morfoloji)"]
    C --> D["İnceltme (Thinning)"]
    D --> E["Sınır Koordinatları (x_i, y_i)"]
    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#0f3460,color:#fff
    style C fill:#16213e,stroke:#0f3460,color:#fff
    style D fill:#16213e,stroke:#0f3460,color:#fff
    style E fill:#0f3460,stroke:#4cc9f0,color:#fff
```

---

### 2.2. En Küçük Kareler Doğru Uydurma (Least Squares Line Fitting)

Verilen $N$ adet $(x_i, y_i)$ kenar noktasına en uygun $y = mx + c$ doğrusunu uydurmak istediğimizi varsayalım. Buradaki amaç, eğim ($m$) ve kesim noktasını ($c$) saptamaktır.

#### 2.2.1. Dikey Mesafe (Vertical Distance) Minimizasyonu

En klasik yöntem, her noktanın doğruya olan ortalama karesel dikey uzaklığını (*average squared vertical distance*) minimize etmektir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-02.png" alt="Vertical Distance Line Fitting" style="display:flex; border-radius: 5px; justify-content: center; width: 420px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 2: En küçük kareler doğru uydurmada $(x_i, y_i)$ noktasının $y = mx + c$ doğrusuna dikey uzaklığı $|y_i - mx_i - c|$.</em></figcaption>
  </div>
</figure>

Bir $(x_i, y_i)$ noktasının doğruya olan dikey uzaklığı $y_i - m x_i - c$ ile verilir. Buradan ortalama karesel hata enerji (maliyet) fonksiyonu tanımlanır:

$$E = \frac{1}{N} \sum_{i=1}^{N} (y_i - m x_i - c)^2$$

Bu enerjiyi minimize etmek için $m$ ve $c$ parametrelerine göre kısmi türevler alınır ve sıfıra eşitlenir:

$$\frac{\partial E}{\partial m} = 0 \implies \frac{1}{N} \sum_{i=1}^{N} 2(y_i - m x_i - c)(-x_i) = 0 \implies \sum_{i=1}^{N} (y_i - m x_i - c)x_i = 0$$

$$\frac{\partial E}{\partial c} = 0 \implies \frac{1}{N} \sum_{i=1}^{N} 2(y_i - m x_i - c)(-1) = 0 \implies \sum_{i=1}^{N} (y_i - m x_i - c) = 0$$

İkinci denklemden kesim noktası $c$ çekilirse:

$$c = \bar{y} - m\bar{x} \quad \text{burada} \quad \bar{x} = \frac{1}{N}\sum_{i=1}^N x_i, \quad \bar{y} = \frac{1}{N}\sum_{i=1}^N y_i$$

Bu ifade birinci türev denkleminde yerine koyulup düzenlendiğinde, eğim $m$ için analitik kapalı form (*closed-form*) çözüm elde edilir:

$$m = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{N} (x_i - \bar{x})^2}$$

---

#### 2.2.2. Dikey Doğrularda Çökme Problemi (The Vertical Line Failure)

Dikey mesafe minimizasyonu yöntemi, kenar noktaları dikey (düşey) bir doğru oluşturduğunda matematiksel olarak tamamen çöker.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-03.png" alt="Vertical Line Failure Mode" style="display:flex; border-radius: 5px; justify-content: center; width: 400px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 3: Dikey doğru çökme problemi: Dikey hizalanmış noktalar için dikey mesafe minimizasyonu tamamen yanlış yatay bir doğru uydurur.</em></figcaption>
  </div>
</figure>

- **Fiziksel Neden:** Dikey bir doğrunun eğimi sonsuza gider ($m \to \infty$). Paydadaki $\sum (x_i - \bar{x})^2$ terimi sıfıra yaklaşacağından denklem tanımsız hale gelir.
- **Hatalı Davranış:** Enerji fonksiyonu dikey mesafeyi ölçtüğü için, dikey doğru üzerindeki noktaların dikey olarak uydurulan bir dikey doğruya mesafeleri tanımsızdır. Algoritma dikey mesafeleri azaltmak adına, gerçek dikey doğrunun tam aksine dikey mesafeleri sıfırlayan tamamen yanlış yatay bir doğru uydurur.

---

### 2.3. Dik Mesafe Minimizasyonu (Average Squared Perpendicular Distance)

Dikey çizgi tekilliğini ortadan kaldırmak için, çizginin dik normal parametrizasyonu tercih edilir:

$$x \sin\theta - y \cos\theta + \rho = 0$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-05.png" alt="Line Normal Parametrization" style="display:flex; border-radius: 5px; justify-content: center; width: 480px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 4: Doğrunun normal form parametrizasyonu ($\theta, \rho$). $\theta$ normal açısını, $\rho$ orijine olan dik mesafeyi gösterir.</em></figcaption>
  </div>
</figure>

Burada $\theta$ doğrunun yatay eksenle yaptığı açıyı, $\rho$ ise orijine olan en kısa dik uzaklığı temsil eder. Bir $(x_i, y_i)$ noktasının bu doğruya olan dik uzaklığı (*perpendicular distance*) doğrudan şu ifadeye eşittir:

$$r_i = x_i \sin\theta - y_i \cos\theta + \rho$$

Bu uzaklıkların karesinin ortalamasını minimize eden enerji fonksiyonu kurulur:

$$E = \frac{1}{N} \sum_{i=1}^{N} (x_i \sin\theta - y_i \cos\theta + \rho)^2$$

> **Key Insight (İkili Görüntüler ile Matematiksel İlişki):** Dik mesafe minimizasyonu formülasyonu, İkili Görüntü İşleme dersindeki **En Küçük İkinci Moment Ekseni (*axis of minimum second moment*)** hesabı ile matematiksel olarak birebir özdeştir.

Noktalar ikili nesne pikselleri gibi ele alınarak kütle merkezine $(\bar{x}, \bar{y})$ göre şu ikinci momentler hesaplanır:

$$a = \sum_{i=1}^N (x_i - \bar{x})^2, \quad c = \sum_{i=1}^N (y_i - \bar{y})^2, \quad b = 2 \sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})$$

Bu moment sabitleri kullanılarak dikey çizgi tekilliği yaşanmadan $\theta$ ve $\rho$ değerleri kararlı bir şekilde çözülür:

$$\tan(2\theta) = \frac{b}{a - c}$$

$$\rho = \bar{y}\cos\theta - \bar{x}\sin\theta$$

---

### 2.4. Eğri (Polinom) Uydurma ve Overdetermined Sistem Çözümü

Kenar noktaları bir doğru yerine karmaşık bir eğri oluşturuyorsa, örneğin 3. dereceden bir polinom ($y = ax^3 + bx^2 + cx + d$) uydurulmak istenebilir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-04.png" alt="Polynomial Curve Fitting" style="display:flex; border-radius: 5px; justify-content: center; width: 420px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 5: Noktalar kümesine $y = f(x)$ parametrik polinom eğrisinin uydurulması.</em></figcaption>
  </div>
</figure>

Karesel dikey uzaklık enerjisi şu şekilde tanımlanır:

$$E = \frac{1}{N} \sum_{i=1}^{N} (y_i - a x_i^3 - b x_i^2 - c x_i - d)^2$$

Bu enerjinin her bir bilinmeyen katsayıya ($a, b, c, d$) göre türevinin alınıp sıfıra eşitlenmesi hantal bir süreçtir. Bunun yerine sistem, **aşırı belirlenmiş (*over-determined*) doğrusal denklem sistemi** olarak matris formunda ifade edilir.

Her bir $(x_i, y_i)$ noktası polinom denklemine yazılarak $N$ adet denklem elde edilir:

$$\begin{aligned}
y_1 &= a x_1^3 + b x_1^2 + c x_1 + d \\
y_2 &= a x_2^3 + b x_2^2 + c x_2 + d \\
&\ \ \vdots \\
y_N &= a x_N^3 + b x_N^2 + c x_N + d
\end{aligned}$$

Bilinmeyen sayısı $m$ (burada $m=4$: $a, b, c, d$) ve nokta sayısı $N$ olmak üzere ($N > m$), bu sistem vektör-matris formuna dönüştürülür:

$$X a = y$$

$$\begin{bmatrix} 
x_1^3 & x_1^2 & x_1 & 1 \\ 
x_2^3 & x_2^2 & x_2 & 1 \\ 
\vdots & \vdots & \vdots & \vdots \\ 
x_N^3 & x_N^2 & x_N & 1 
\end{bmatrix}_{N \times m} 
\begin{bmatrix} a \\ b \\ c \\ d \end{bmatrix}_{m \times 1} = 
\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix}_{N \times 1}$$

Burada $X$ girdi matrisi ($N \times m$) kare bir matris olmadığından doğrudan matris tersi (*inverse*) alınamaz. En küçük kareler çözümünü elde etmek için denklem her iki taraftan $X^T$ (transpoz) ile çarpılarak $m \times m$ boyutlu kare bir matrise dönüştürülür:

$$X^T X a = X^T y \implies a = (X^T X)^{-1} X^T y$$

Bu denklemdeki $X^+ = (X^T X)^{-1} X^T$ ifadesine **Sözde Evrik (Moore-Penrose Pseudo-Inverse)** adı verilir. Bu yaklaşım her dereceden polinom eğri uydurma problemleri için genel ve son derece kararlı bir çözümdür.

---

## 3. Aktif Konturlar (Active Contours / Snakes)

**Aktif Konturlar (Snakes)**, bir nesnenin sınırlarını saptamak amacıyla, nesnenin etrafına kabaca çizilen bir başlangıç konturunun zaman içinde iteratif olarak büzülüp şekil değiştirerek nesnenin gerçek sınırlarına bir lastik bant gibi oturmasını sağlayan dinamik ve güçlü bir deformasyon yöntemidir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-06.png" alt="Deformable Boundaries Examples" style="display:flex; border-radius: 5px; justify-content: center; width: 620px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 6: Deforme olabilen sınırlar: Zaman içinde şekil değiştiren dudak hareketi ve bakış açısına göre değişen araç silüeti.</em></figcaption>
  </div>
</figure>

### 3.1. Konturun Ayrık Temsili (Contour Representation)

Kontur, sürekli bir eğrinin ayrıklaştırılmasıyla elde edilen ve birbirine eşit uzunluktaki doğru parçalarıyla bağlı $N$ adet kontrol noktasından (*control points*) oluşan sıralı bir liste şeklinde temsil edilir:

$$v_i = (x_i, y_i) \quad \text{burada} \quad i = 0, 1, 2, \dots, N-1$$

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-08.png" alt="Contour Representation" style="display:flex; border-radius: 5px; justify-content: center; width: 480px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 7: Konturun $N$ adet kontrol noktası $v_i = (x_i, y_i)$ ile ayrık olarak temsil edilmesi.</em></figcaption>
  </div>
</figure>

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-07.png" alt="Initial Contour around Quarter Coin" style="display:flex; border-radius: 5px; justify-content: center; width: 360px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 8: Madeni para etrafında ilklendirilen (kabaca çizilen) başlangıç konturu ve kontrol noktaları.</em></figcaption>
  </div>
</figure>

---

### 3.2. Enerji Formülasyonu ve Kuvvetler

Konturu nesne sınırına doğru hareket ettiren (dış kuvvetler) ve aynı zamanda pürüzsüz yapısını korumasını sağlayan (iç kuvvetler) iki temel enerji bileşeni tanımlanır.

#### 3.2.1. Kontur Enerjisi ($E_{contour}$ - İç Kuvvetler)

Konturun gürültüye kapılıp ani kıvrılmalar, düğümler yapmasını engellemek, yani pürüzsüz kalmasını sağlamak için iç bükülme enerjisi (*internal energy*) tanımlanır. Bu enerji iki fiziksel terimden oluşur:

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-09.png" alt="Physical Intuition of Internal Energy" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 9: İç enerjilerin fiziksel sezgisi: Esneklik bir lastik bant (rubber band) gibi büzülmeyi, Pürüzsüzlük ise metal şerit (metal strip) gibi yumuşak kıvrılmayı temsil eder.</em></figcaption>
  </div>
</figure>

1. **Esneklik (Elasticity - $E_{elastic}$):** Konturun bir lastik bant (*rubber band*) gibi büzülmesini ve kontrol noktaları arasındaki mesafelerin minimumda tutulmasını sağlar. Sürekli uzayda birinci türevin karesine ($|\frac{\partial v}{\partial s}|^2$) karşılık gelirken, ayrık uzayda ardışık kontrol noktaları arasındaki karesel mesafe ile hesaplanır:

$$E_{elastic} = \sum_{i=0}^{N-1} |v_{i+1} - v_i|^2$$

2. **Pürüzsüzlük (Smoothness - $E_{smooth}$):** Konturun bükülme miktarını (*curvature*) minimize ederek ani yön değişimlerini engeller ve pürüzsüz bir metal şerit (*metal strip*) gibi davranmasını sağlar. Sürekli uzayda ikinci türevin karesine ($|\frac{\partial^2 v}{\partial s^2}|^2$) karşılık gelirken, ayrık uzayda farkların ikincil farkı ile hesaplanır:

$$E_{smooth} = \sum_{i=0}^{N-1} |v_{i+1} - 2v_i + v_{i-1}|^2$$

Bu iki terim $\alpha$ ve $\beta$ ağırlık katsayılarıyla birleştirilerek iç kontur enerjisi ($E_{contour}$) oluşturulur:

$$E_{contour} = \alpha E_{elastic} + \beta E_{smooth} = \alpha \sum_{i=0}^{N-1} |v_{i+1} - v_i|^2 + \beta \sum_{i=0}^{N-1} |v_{i+1} - 2v_i + v_{i-1}|^2$$

---

#### 3.2.2. Görüntü Enerjisi ($E_{image}$ - Dış Kuvvetler)

Konturu yüksek gradyanlı nesne sınırlarına çekmek için görüntünün gradyan büyüklüğünün karesi ($|\nabla I|^2$) kullanılır. Ancak kontur nesneye uzaksa, o konumlarda gradyan değerleri sıfıra yakın olacağından kontura hiçbir çekim kuvveti uygulanamaz.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-10.png" alt="Blurred Gradient Magnitude Potential Field" style="display:flex; border-radius: 5px; justify-content: center; width: 620px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 10: Görüntü Enerjisi: Orijinal kontur (sol), ham gradyan büyüklüğü $\|\nabla I\|^2$ (orta) ve Gauss filtresi ile bulanıklaştırılmış $\|\nabla G_\sigma * I\|^2$ potansiyel çekim alanı (sağ).</em></figcaption>
  </div>
</figure>

> **Bulanıklaştırma (Blurring) Hilesi:** Gradyan haritası geniş standart sapmalı bir Gauss filtresi ($G_\sigma$) ile bulanıklaştırılarak (*blurred*) geniş bir çekim alanı veya potansiyel kuvvet alanı (*potential/force field*) oluşturulur. Bu sayede kontur uzak konumda olsa bile merkeze/sınıra doğru çekilir.

Gradyan toplamını maksimize etmek, negatifini minimize etmeye eşdeğer olduğundan dış görüntü enerjisi şu şekilde tanımlanır:

$$E_{image} = - \sum_{i=0}^{N-1} |\nabla (G_\sigma * I(v_i))|^2$$

---

#### 3.2.3. Toplam Enerji ($E_{total}$)

Konturun optimize etmeye çalıştığı nihai enerji, dış ve iç enerjilerin toplamıdır:

$$E_{total} = E_{image} + E_{contour}$$

---

### 3.3. Deformasyon Algoritması (Greedy Algorithm)

Toplam enerjiyi minimize etmek için pratik ve hızlı bir **açgözlü (*greedy*) algoritma** uygulanır:

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-12.png" alt="Greedy Algorithm Local Window Search" style="display:flex; border-radius: 5px; justify-content: center; width: 420px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 11: Greedy Algoritmasında her bir $v_i$ kontrol noktası için etrafındaki $W$ yerel arama penceresindeki (mavi kareler) konumların test edilmesi.</em></figcaption>
  </div>
</figure>

1. **Düzgün Yeniden Örnekleme (Uniform Re-sampling):** Kontur üzerindeki kontrol noktaları arasındaki mesafeler eşitlenecek şekilde yeniden örneklenir (*re-sampling*).
   - *Kritik Önemi:* Eğer bu adım her iterasyon başında tekrarlanmazsa, esneklik kuvvetleri yüzünden kontrol noktaları belirli bölgelerde yığılır, düğümlenir ve kontur yapısı bozulur.
2. **Lokal Arama ve Taşıma:** Her bir $v_i$ kontrol noktası için etrafındaki küçük bir $W$ arama penceresindeki (örneğin $3 \times 3$ veya $5 \times 5$ piksel) tüm komşu konumlar test edilir. Nokta, yerel $E_{total}$ enerjisini minimum yapan yeni konuma taşınır.
3. **Durdurma Kriteri:** Eğer tüm noktaların o iterasyondaki hareket miktarlarının toplamı belirlenen çok küçük bir $\epsilon$ eşik değerinden küçükse algoritma durdurulur (kontur dengeye ulaşmıştır). Aksi takdirde Adım 1'e dönülür.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-13.png" alt="Failure without Uniform Resampling" style="display:flex; border-radius: 5px; justify-content: center; width: 400px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 12: Düzgün yeniden örnekleme yapılmadığında esneklik kuvvetleri nedeniyle noktaların düğümlenmesi ve kontur çökme hatası.</em></figcaption>
  </div>
</figure>

---

### 3.4. Parametre Analizi ve İleri Yöntemler

#### 3.4.1. $\alpha$ Parametresinin Etkisi

Esneklik katsayısı $\alpha$, konturun büzülme şiddetini belirler.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-fitting-lines-and-active-contours-11.png" alt="Effect of Alpha Parameter" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Şekil 13: Yan yana iki madeni para örneğinde $\alpha$ parametresinin etkisi. Büyük $\alpha$ konturu iki para arasındaki dar boşluğa büzüştürürken, küçük $\alpha$ daha gevşek bir hat çizer.</em></figcaption>
  </div>
</figure>

- **Büyük $\alpha$:** Kontur yüksek bir esneklik gerilimi altındadır; adeta sıkı bir lastik bant gibi davranarak iki nesne arasındaki dar girintilere büzülür.
- **Küçük $\alpha$:** Esneklik gerilimi düşüktür; kontur girintilere girmek yerine nesneleri dışarıdan daha gevşek sarmalar.

#### 3.4.2. Sınır Koşulları ve İleri Modeller

- **İlklendirme Duyarlılığı (Initialization Sensitivity):** Aktif konturlar iyi bir başlangıç tahminine (*initialization*) ihtiyaç duyar. Eğer başlangıç eğrisi nesneye çok uzak çizilirse, bulanıklaştırılmış gradyanların çekim alanının dışında kalır ve alakasız gürültülere veya başka nesnelere takılır.
- **Balonlaşma Kuvvetleri (Ballooning Forces):** Klasik model konturu içe doğru büzerken (*lastik bant etkisi*), dış kuvvete bir balonlama terimi eklenerek konturun nesnenin içinden dış sınırlarına doğru genişlemesi (*ballooning*) sağlanabilir.
- **Önsel Şekil Modelleri (Prior Shape Models):** Şekli önceden bilinen nesneler için (örneğin kalp veya göz), hedef şekilden sapmaları cezalandıran önsel bir şekil enerjisi ($E_{prior}$) toplam enerjiye eklenebilir.
