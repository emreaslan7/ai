# Genel Bakış, Tarihçe ve Görüntü Sensör Türleri

<!-- toc -->

## 1. Genel Bakış

Görüntü algılama (image sensing), üç boyutlu (3D) bir sahneden yayılan veya yansıyan elektromanyetik radyasyonun (ışığın) yakalanarak iki boyutlu (2D) kalıcı ve ölçülebilir bir temsil biçimine dönüştürülmesi işlemidir. Optik sistemler (mercekler ve diyaframlar) projeksiyonun *geometrisini* yönetirken, görüntü algılama mekanizması *fotometrik dönüşümü* yönetir — yani gelen foton akısını filmdeki kimyasal indirgenmeye veya silisyumdaki elektrik yüküne dönüştürür.

Görüntü sensörlerinin evrimini ve fiziğini anlamak, bilgisayarlı görü (computer vision) disiplininin temelini oluşturur: Dijital pikseller üzerinde çalışan her algoritma, alt katmandaki sensör mimarisinin optoelektronik özelliklerine, örnekleme sınırlarına, dinamik aralığına ve gürültü karakteristiklerine doğrudan bağımlıdır.

> **Temel Sezgi (Key Insight):** Optik düzenek ışık ışınlarının düzlemde *nereye* düşeceğini belirler; algılama mekanizması ise foton enerjisinin *nasıl* sayılabilir bir sinyale (yük veya voltaj) dönüştürüleceğini yönetir.

---

## 2. Görüntülemenin Kısa Tarihçesi

Işığı yakalama ve fiziksel dünyayı iki boyutlu bir yüzeye yansıtma yolculuğu, pasif optik projeksiyondan kimyasal depolamaya ve nihayetinde dijital silisyum mimarilerine uzanan asırlık bilimsel ve sanatsal bir evrimi kapsar.

```mermaid
flowchart TD
    T1["M.Ö. 500 — İğne Deliği Kamera<br/>(Camera Obscura)"] --> T2["17. Yüzyıl — Mercek Entegrasyonu<br/>ve Ayna Katlama"]
    T2 --> T3["1830'lar — Kimyasal Film Devrimi<br/>(Dagerotip)"]
    T3 --> T4["1970'ler — Silisyum Görüntü Dedektörü<br/>(Yeniden Kullanılabilir Sensör)"]
    T4 --> T5["2000'ler-Günümüz — Akıllı Kameralar<br/>ve Wafer Entegrasyonu"]

    style T1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style T2 fill:#16213e,stroke:#4cc9f0,color:#fff
    style T3 fill:#0f3460,stroke:#f72585,color:#fff
    style T4 fill:#06d6a0,stroke:#111,color:#000
    style T5 fill:#118ab2,stroke:#fff,color:#fff
```

### 2.1 İğne Deliği Kamera (Camera Obscura)

Görüntü oluşumunun temel kavramları M.Ö. 500 yıllarına, Çinli filozofların iğne deliği kamera ilkelerini belgelemesine kadar uzanır. M.S. 1000 civarında, Arap bilim insanı İbn-i Heysem (Alhazen), iğne deliği kameranın optik özelliklerini ve geometrik projeksiyonunu titizlikle analiz etmiştir.

Konseptin Batı'da, özellikle sanatçılar arasında yaygınlaşması 16. yüzyılı bulmuştur. Felemenkli matematikçi Gemma Frisius'un 1544 tarihli çiziminde gösterildiği gibi:
1. Karanlık bir odanın duvarındaki minik bir iğne deliği, 3D bir sahneyi karşı duvara yansıtarak ters dönmüş 2D bir görüntü oluşturur.
2. Sanatçı camera obscura odasına girerek duvardaki projeksiyonun üzerinden çizebilir ve 3D sahnenin geometrik olarak son derece doğru çizimlerini elde edebilirdi.

> **Optik Kısıt:** İğne deliği kamera sonsuz alan derinliğinde son derece keskin görüntüler üretse de, açıklığı matematiksel olarak çok küçük olduğu için çok az foton toplar. Sonuçta oluşan projeksiyonlar son derece karanlıktır ve gözün karanlığa uyum sağlaması gerekir.

### 2.2 Mercek ve Ayna Entegrasyonu

İğne deliğinin foton yetersizliğini (photon starvation) çözmek amacıyla 17. yüzyıl tasarımcıları, minik iğne deliği yerine kırıcı bir dışbükey (konveks) mercek yerleştirdiler. Mercek, çok daha geniş bir ışık konisini odaklayarak belirgin şekilde daha parlak görüntüler oluşturdu.

18. yüzyılda optomekanik tasarımlar sanatçı ergonomisine odaklandı:
- Mercek tarafından yansıtılan dikey ışık konusu $45^\circ$'lik bir ayna ile katlandı.
- Bu düzenek ışığı yukarıya, yatay ve yarı saydam bir aydınger kâğıdına yönlendirdi.
- Sanatçı rahatça oturup aşağıya bakarak sahnenin üzerinden çizebiliyordu. Bu optomekanik mimari, daha sonra Tek Mercekli Yansımalı (SLR) vizör sistemlerine ilham vermiştir.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-01.png" alt="18. Yüzyıl Aynalı/Mercekli Box Camera Obscura" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>18. Yüzyıl Aynalı/Mercekli Box Camera Obscura: Merceğin oluşturduğu görüntüyü 45 derecelik bir ayna ile yatay buzlu cama katlayarak ressamların çizimini kolaylaştıran optomekanik kutu tasarımı.</em></figcaption>
  </div>
</figure>

### 2.3 Kimyasal Film Devrimi

Görüntülemedeki en köklü kültürel sıçrama 1830'larda Louis Daguerre'in **Dagerotip (Daguerreotype)** kamerasını icat etmesiyle gerçekleşti. 1837'de çekilen natürmort fotoğraflar, bir sahnenin insan sanatçıyı aradan çıkararak tek bir düğmeye basışla kalıcı bir kimyasal ortama kaydedilebileceğini kanıtladı.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-02.png" alt="Louis Daguerre - Still Life (1837)" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Louis Daguerre - Still Life (1837): Daguerreotype kamera ile çekilen ve insanlık tarihinin ilk kalıcı kimyasal film görüntülerinden biri olan alçı büst ve obje natürmortu.</em></figcaption>
  </div>
</figure>

#### Siyah-Beyaz Filmin Kimyasal Süreci
1. **Emülsiyon:** Film, ışığa duyarlı gümüş halojenür kristalleri ($\text{AgX}$, burada $\text{X} = \text{Br, Cl, I}$) içeren mikroskobik bir katmanla kaplanır.
2. **Pozlama (Exposure):** Foton emilimi, gümüş iyonlarının metallic gümüşe indirgenmesini tetikler:
   $$\text{Ag}^+ + e^- \xrightarrow{h\nu} \text{Ag}^0$$
   Toplam pozlama enerjisi $E$, karşılıklılık yasasına (reciprocity law) uyar:
   $$\text{Pozlama } (E) \propto \text{Işınım (Irradiance } I) \times \text{Entegrasyon Süresi } (T)$$
3. **Banyo / Geliştirme (Development):** Kimyasal banyo, bu gizil (latent) gümüş görüntüsünü büyüterek kararlı, yüksek çözünürlüklü bir fotoğraf negatifine dönüştürür.

#### Renkli Filme Geçiş (1880'ler)
Tüm görünür spektrumu yakalamak daha karmaşık bir emülsiyon kimyası gerektirdi. 1887'de Louis Ducos du Hauron; Kırmızı, Yeşil ve Mavi pigmentler içeren boya bağlayıcılı katmanları gümüş halojenür ile üst üste istifleyerek ilk renkli fotoğrafları elde etti.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-03.png" alt="Louis Ducos du Hauron - Angoulême Manzarası (1877/1887)" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Louis Ducos du Hauron - Angoulême Manzarası (1877/1887): Üç renkli (kırmızı, yeşil, mavi) emülsiyon ve boya bağlayıcı katmanlarla çekilen ilk renkli manzara fotoğrafı.</em></figcaption>
  </div>
</figure>

1920'lere gelindiğinde Ernemann gibi tüketici kameraları *"Görebildiğin her şeyi fotoğraflayabilirsin"* sloganıyla kitle pazarına girdi ve görsel kaydı insan ifadesinin evrensel bir aracı haline getirdi.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-04.png" alt="Ernemann Katlanabilir Plaka Film Kamerası" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Ernemann Katlanabilir Plaka Film Kamerası: 1920'lerde mass-market tüketici fotoğrafçılığını başlatan ve "Gördüğünü fotoğraflayabilirsin" reklamıyla sunulan ikonik cihaz.</em></figcaption>
  </div>
</figure>

### 2.4 Silisyum Görüntü Dedektörü (Silicon Detector)

Kimyasal film görsel kültürü devrimcileştirmiş olsa da, tek kullanımlık bir sarf malzemesi olması en büyük kısıtıydı. 1970'lerde silisyum görüntü dedektörünün icadı bu paradigmayı kökten değiştirdi:

- Kimyasal filmin aksine, silisyum sensör kimyasal banyo gerektirmeden sonsuz sayıda görüntü dizisi yakalayabilen **yeniden kullanılabilir bir katı hal (solid-state) cihazıdır**.
- Silisyum üretiminin tüketici seviyesine ulaşması yaklaşık 20 yıl sürdü ve 1990'ların başında Nikon COOLPIX gibi ilk dijital kameralar piyasaya çıktı.
- Bu ilk cihazlar $640 \times 480$ piksel ($\approx 0.3\text{ MP}$) çözünürlük sunuyor, yüksek güç tüketiyor ve hızlı depolamadan yoksun bulunuyordu; ancak dijital görüntü işlemenin geleceğini kanıtladı.

### 2.5 Akıllı Telefon Kameraları ve AI Katalizörü

20. yüzyılın sonları ve 21. yüzyılın başlarında kamera modüllerinin cep telefonlarına entegre edilmesi, benzeri görülmemiş bir minyatürleşme ve optik mühendisliği hamlesi başlattı.

- 2007'de akıllı telefonların doğuşu ikinci dijital kamera devrimini tetikledi.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-05.png" alt="Apple iPhone 1 (2007) Arka Gövde Görseli" style="display:flex; border-radius: 5px; justify-content: center; width: 200px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Apple iPhone 1 (2007) Arka Gövde Görseli: Tüketici elektroniğinde kamera minyatürleşmesinin miladı sayılan ve bilgisayarlı görünün gelişimini tetikleyen ilk iPhone tasarımı.</em></figcaption>
  </div>
</figure>

- Bu mobil kamera patlaması petabaytlarca görsel veri üreten küresel iletişim platformlarını doğurdu.
- En önemlisi, bu devasa dijital görüntü akışı, modern bilgisayarlı görü ve derin öğrenme algoritmalarının temel veri kümesini ve hesaplamalı katalizörünü oluşturdu.

### 2.6 Yüzyıllık Karşılaştırma: Kodak Brownie vs. Modern Akıllı Telefon Kamerası

| Özellik / Parametre | Kodak Brownie Model 1 (1900) | Modern Akıllı Telefon Kamera Modülü |
| :--- | :--- | :--- |
| **Satış Fiyatı** | 1.00$ USD (Enflasyon ayarlı ~30$ USD) | Kitlesel üretimle optimize edilmiş maliyet |
| **Optik Geometri** | Tekil küresel cam mercek elemanı | Çok elemanlı, ultra ince asferik kalıplanmış plastik/cam mercekler |
| **Odaklama Mekanizması** | Sabit odaklı (Fixed-focus) sistem | Mikron hassasiyetli ses bobini motoru (VCM) ile dinamik otofokus |
| **Diyafram Kontrolü** | Farklı delik çaplarına sahip kayar metal plaka | Minyatür diyafram dizileri veya sabit düşük F-sayılı açıklıklar ($f/1.5 - f/1.8$) |
| **Vizör ve Geri Bildirim** | Köşede küçük yansıtıcı ayna (sensör beslemesi yok) | Canlı elektronik sensör akışını gösteren gerçek zamanlı dijital ekran |
| **Kayıt Ortamı / Gecikme** | Gümüş halojenür rulo film; postayla gönderim; haftalarca gecikme | Silisyum sensör; dahili ISP; anlık görselleştirme ve sıfır deklanşör gecikmesi |

### 2.7 Gelecek Vizyonu: Wafer Seviyesinde Entegrasyon (Wafer-Scale Integration)

Sensör tasarımındaki bir sonraki paradigma kayması **Optics-on-Wafer** ve **3D-Stacked Sensor** teknolojisidir:

```mermaid
flowchart TD
    A["1. Kırıcı Mikromercek Dizisi<br/>(Doğrudan yarı iletken wafer üzerinde büyütülür)"] --> B["2. Renk Filtresi ve Fotodiyot Dizisi<br/>(Üst Silisyum Algılama Katmanı)"]
    B --> C["3. 3D İstiflenmiş Mikro-Elektronik Taban<br/>(Doğrudan Hibrit Bağlama)"]
    C --> D["4. Yonga Üstü Nöral İşlem Birimi (NPU)<br/>ve ISP Yürütme Motoru"]
    
    style A fill:#1a1a2e,stroke:#4cc9f0,color:#fff
    style B fill:#16213e,stroke:#e94560,color:#fff
    style C fill:#0f3460,stroke:#f72585,color:#fff
    style D fill:#06d6a0,stroke:#111,color:#000
```

- Tamamlanmış sensörün üzerine ayrı plastik mercekler monte etmek yerine, kırıcı mercek elemanları dökümhanede doğrudan silisyum wafer üzerinde büyütülür.
- Algılama katmanının altına 3D istiflenmiş elektronik devreler doğrudan entegre edilir.
- Bu mimari; Görüntü Sensörünü, Renk Filtresini, Mikromercekleri ve dijital mikro-nöral işlemcileri tek bir birleşik yongada toplar — kamerayı pasif bir yakalama cihazından otonom bir yonga üstü bilgisayarlı görü sistemine dönüştürür.

---

## 3. Görüntü Sensör Türleri ve Katı Hal Fiziği

### 3.1 Silisyum Foto-Dönüşümünün Fiziği

Dijital görüntü algılamanın temel mekanizması, kristal silisyumun ($\text{Si}$) optoelektronik özelliklerine dayanır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-06.png" alt="Silisyum Foto-Konversiyon Fiziği Şeması" style="display:flex; border-radius: 5px; justify-content: center; width: 450px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Silisyum Foto-Konversiyon Fiziği Şeması: Gelen fotonun silikon atomuna çarparak valans elektronunu iletim bandına uyarmasını ve elektron-delik çifti (electron-hole pair) oluşturmasını gösteren şema.</em></figcaption>
  </div>
</figure>

```mermaid
flowchart TD
    A["Gelen Foton<br/>(Enerji E = hν ≥ E_g)"] -->|"Silisyum Kristaline Çarpar"| B["Silisyum Atomu<br/>(Bant Aralığı E_g ≈ 1.11 eV, 300K)"]
    B --> C["Valans Elektronu İletkenlik Bandına Uyarılır"]
    C --> D["Serbest Elektron (e⁻) Üretilir"]
    C --> E["Pozitif Yüklü Boşluk / Hole (h⁺) Oluşur"]
    
    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#4cc9f0,color:#fff
    style C fill:#0f3460,stroke:#f72585,color:#fff
    style D fill:#06d6a0,stroke:#111,color:#000
    style E fill:#ffd166,stroke:#111,color:#000
```

1. **Bant Aralığı Enerjisi ($E_g$):** Silisyumun bant aralığı oda sıcaklığında ($300\text{ K}$) yaklaşık $E_g \approx 1.11\text{ eV}$'dir. $E = h\nu \ge E_g$ enerjisine sahip gelen bir foton silisyum kristaline çarptığında, bir valans elektronunu iletkenlik bandına uyarır.
2. **Elektron-Boşluk Çifti Üretimi:** Bu uyarılma serbest bir iletkenlik elektronu ($e^-$) oluşturur ve geride pozitif yüklü bir boşluk ($h^+$) bırakır.
3. **Kuantum DENGESİ:** Sürekli aydınlatma altında, gelen foton akısı ile üretilen elektron akısı arasında bir kararlı durum kurulur. Biriken bu elektron yükünün ölçülmesi, o konuma düşen ışık yoğunluğunu nicelleştirmemizi sağlar:
   $$Q = \int_{0}^{T} \frac{\eta \cdot q \cdot P(t)}{h\nu} \, dt$$
   burada $\eta$ kuantum verimliliği, $q$ elektron yükü, $P(t)$ optik güç ve $T$ pozlama süresidir.

> **Mühendislik Zorluğu:** Silisyum optik-elektronsal dönüşümü doğal olarak gerçekleştirir. Temel mühendislik zorluğu, milyonlarca pikseldeki bu hassas elektron paketlerini gürültü, sinyal bozulması veya çapraz etkileşim (cross-talk) olmadan okumaktır.

### 3.2 Minyatürleşme Sınırları ve Moore Yasası

Modern yüksek yoğunluklu sensörler, piksel aralığı $1.25\ \mu\text{m}$'ye kadar düşen onlarca megapikseli küçücük mobil alanlara sığdırır. Ancak piksel küçültme, ışığın kırınım fiziği nedeniyle Moore Yasasını sonsuza kadar takip edemez:

- **Görünür Dalga Boyu Spektrumu:** Görünür ışık $\lambda \approx 400\text{ nm}$ (mor) ile $\lambda \approx 700\text{ nm}$ (kırmızı) arasında değişir.
- **Kırınım Sınırı (Diffraction Limit):** Piksel boyutu $d$ yaklaşık yarım mikrometreye ($d \approx 0.5\ \mu\text{m}$) düştüğünde, ışığın dalga boyu mertebesine ulaşır:
  $$d_{\text{sınır}} \approx \frac{\lambda}{2}$$
- Bu sınırın altında optik kırınım (diffraction) baskın hale gelir. Işık dalgaları piksel sınırlarından bükülerek komşu pikseller arasında şiddetli optik çapraz etkileşime (cross-talk) yol açar ve fiziksel alan ne kadar küçültülürse küçültülsün gerçek optik çözünürlük artışını engeller.

> **Ana Çıkarım:** Kırınım sınırının ötesinde çözünürlüğü artırmak için mühendisler bireysel pikselleri küçültmek yerine silisyum yonganın fiziksel alanını büyütmek zorundadır.

### 3.3 CCD (Charge Coupled Device) Mimarisi

1969 yılında Willard Boyle ve George E. Smith tarafından icat edilen **CCD (Charge-Coupled Device)** mimarisi, bir analog kaydırmalı yazmaç (shift register) gibi çalışır.

```mermaid
flowchart TD
    subgraph Matrix ["Fotodiyot Dizisi (Potansiyel Kuyuları)"]
        P11["Piksel (1,1)<br/>Yük Paketi e⁻"] --- P12["Piksel (1,2)<br/>Yük Paketi e⁻"]
        P21["Piksel (2,1)<br/>Yük Paketi e⁻"] --- P22["Piksel (2,2)<br/>Yük Paketi e⁻"]
    end
    
    Matrix -->|"Satır Satır Dikey Aktarım<br/>(Çok Fazlı Elektrik Alanları)"| VSR["Dikey Taşıma Yazmacı"]
    VSR -->|"Seri Satır Transferi"| HSR["Yatay Kaydırma Yazmacı"]
    HSR -->|"Piksel Piksel Kaydırma"| AMP["Köşedeki Tekil Yük-Voltaj<br/>Yükselteci (Amplifier)"]
    AMP -->|"Analog Voltaj Sinyali"| ADC["Yonga Dışı Analog-Dijital<br/>Dönüştürücü (ADC)"]
    ADC --> OUT["Dijital Piksel Akışı"]
    
    style Matrix fill:#1a1a2e,stroke:#4cc9f0,color:#fff
    style VSR fill:#16213e,stroke:#e94560,color:#fff
    style HSR fill:#0f3460,stroke:#f72585,color:#fff
    style AMP fill:#06d6a0,stroke:#111,color:#000
    style ADC fill:#118ab2,stroke:#fff,color:#fff
```

#### Okuma Mekanizması: Kovalı Taşıma (Bucket Brigade)
1. **Potansiyel Kuyuları:** Her piksel, pozlama süresince foto-üretilmiş elektronları biriktiren bir potansiyel kuyusu (fotodiyot) olarak görev yapar.
2. **Dikey Satır Aktarımı:** Pozlama tamamlandığında yükler piksel içinde voltaja dönüştürülmez. Elektrot kapılarına uygulanan çok fazlı saat voltajları, tüm yük satırlarını adım adım altındaki potansiyel kuyularına kaydırır.
3. **Yatay Kaydırma ve Yükseltme:** En alt satır yatay kaydırma yazmacına girer ve yükler her defasında tek bir piksel kaydırılarak yonganın köşesindeki yüksek hassasiyetli yük-voltaj yükseltecine iletilir.
4. **Dijitalleştirme:** Köşedeki yükselteç her yük paketini voltaj sinyaline dönüştürür ve bu sinyal yonga dışındaki bir ADC tarafından dijitalleştirilir.

> **Kovalı Taşıma Benzetmesi:** CCD yük transferi, yangını söndürmek için elden ele su kovası taşıyan insan dizisine benzer. Tek bir çıkış yükselteci kullandığı için pikseller arası mükemmel birörnekliğe (uniformity) ve düşük gürültüye sahiptir; ancak yüksek güç tüketimi, yavaş okuma hızı ve blooming duyarlılığı en büyük dezavantajlarıdır.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-07.png" alt="CCD Satır Kaydırma Bucket Brigade Şeması" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>CCD Satır Kaydırma "Bucket Brigade" Şeması: Potansiyel kuyulardaki elektron paketlerinin elektrot elektrik alanları yardımıyla satır satır aşağı, ardından yatay olarak köşedeki amplifikatöre aktarım şeması.</em></figcaption>
  </div>
</figure>

### 3.4 CMOS (Complementary Metal-Oxide Semiconductor) Mimarisi

**CMOS Aktif Piksel Sensörü (APS)** modern dijital görüntülemenin baskın mimarisidir.

```mermaid
flowchart TD
    subgraph Pixel ["Tekil Aktif Piksel Devresi (3T / 4T APS Mimarisi)"]
        PD["Fotodiyot Kuyusu (Foto-Dönüşüm)"] --> TG["Aktarım Kapısı (TG)"]
        TG --> FD["Yüzen Difüzyon (Yerel Yük Depolama)"]
        FD --> SF["Yükselteç Tranzistörü (Source Follower)"]
    end
    
    Pixel --> BUS["Doğrudan Sütun Veri Yolu Hatları<br/>(Rastgele Piksel Adresleme)"]
    BUS --> ADC["Sütun-Paralel ADC Dizisi<br/>(Paralel Dijitalleştirme)"]
    ADC --> OUT["Dijital Görüntü Akışı / ROI Erişimi"]

    style Pixel fill:#1a1a2e,stroke:#e94560,color:#fff
    style PD fill:#16213e,stroke:#4cc9f0,color:#fff
    style TG fill:#0f3460,stroke:#4cc9f0,color:#fff
    style FD fill:#f72585,stroke:#fff,color:#fff
    style SF fill:#06d6a0,stroke:#111,color:#000
    style BUS fill:#118ab2,stroke:#fff,color:#fff
    style ADC fill:#7209b7,stroke:#fff,color:#fff
    style OUT fill:#06d6a0,stroke:#fff,color:#000
```

#### Okuma Mekanizması: Yerel Dönüşüm ve Rastgele Erişim
- **Piksel İçi Yük Dönüşümü:** CCD'lerin aksine, her bir CMOS pikseli kendi foto-diyodunun hemen yanında kendi yük-voltaj dönüştürücü devresine (genellikle 3 veya 4 tranzistörlü aktif piksel tasarımı) sahiptir.
- **Doğrudan Adreslenebilirlik:** CMOS sensörler satır seçme ve sütun okuma hatlarını kullanarak sistem RAM'ine benzer şekilde rastgele piksel adreslemeye izin verir.
- **İlgi Bölgesi (Region of Interest - ROI):** Bu mimari, sensörün tüm çerçeveyi okumak yerine sadece belirlenen alt pencereleri (ROI) aşırı yüksek kare hızlarında okumasını sağlar.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-08.png" alt="CMOS Aktif Piksel Okuma Şeması" style="display:flex; border-radius: 5px; justify-content: center; width: 550px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>CMOS Aktif Piksel Okuma Şeması: Her pikselin yanında kendi elektron-voltaj dönüştürücü devresinin bulunduğu ve veri yolu (bus lines) ile adrese dayalı doğrudan piksel okuma tasarımı.</em></figcaption>
  </div>
</figure>

| Mimari Karşılaştırması | CCD (Charge-Coupled Device) | CMOS (Active-Pixel Sensor) |
| :--- | :--- | :--- |
| **Yük Dönüşümü** | Piksel dışı (Köşede tek yükselteç) | Piksel içi (Her pikselde tranzistör) |
| **Okuma Türü** | Seri yük aktarımı ("Kovalı Taşıma") | Paralel voltaj okuma (Rastgele erişim) |
| **Güç Tüketimi** | Yüksek (Çok fazlı yüksek voltaj saatleri) | Düşük (Standart CMOS dijital voltajı) |
| **Okuma Hızı** | Seri aktarım darboğazı nedeniyle sınırlı | Son derece yüksek (Sütun-paralel ADC'ler) |
| **Dolgu Faktörü (Fill Factor)** | $\approx \%100$ (Piksel içi tranzistör yok) | Düşük (Tranzistörler piksel alanını kaplar) |

### 3.5 Mikro-Optik: Mikromercek Dizisi (Microlens Array)

CMOS sensörlerde piksel içi tranzistör devrelerinin neden olduğu dolgu faktörü (fill factor) kaybını telafi etmek amacıyla üreticiler, sensör yüzeyinin üzerine bir **Mikromercek Dizisi** entegre ederler.

```mermaid
flowchart TD
    L1["Ana Kamera Merceğinden Gelen Işık Işınları"] --> L2["Kavisli Organik Mikromercek Dizisi"]
    L2 -->|"Foton Konisini Odakla"| L3["Renk Filtresi Katmanı (Bayer RGB Boyası)"]
    L3 --> L4["Metal Bağlantı ve İletken Katman (Işığı Engelleyen Yollar)"]
    L4 -->|"Işığı Hassas Boşluğa Yönlendir"| L5["Aktif Silisyum Fotodiyot Penceresi"]
    
    style L1 fill:#1a1a2e,stroke:#888,color:#fff
    style L2 fill:#16213e,stroke:#4cc9f0,color:#fff
    style L3 fill:#0f3460,stroke:#f72585,color:#fff
    style L4 fill:#e94560,stroke:#fff,color:#fff
    style L5 fill:#06d6a0,stroke:#111,color:#000
```

- **Çalışma Prensibi:** Her pikselin üzerine minik kavisli organik bir mikromercek yerleştirilir.
- **Foton Hunileme (Photon Funneling):** Işık ışınlarının duyarsız tranzistör yollarına çarpmasına izin vermek yerine mikromercek, piksel alanındaki tüm fotonları toplayıp kırarak aktif fotodiyot alanına odaklar.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-09.png" alt="Mikromercek ve Filtre Dizilimi 3D Modeli" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Mikromercek ve Filtre Dizilimi 3D Modeli: Silikon taban üzerindeki fotodiyotların üzerine yerleştirilen Bayer filtre mozaiği ve en üstteki organik ışık toplama mikromerceklerinin (microlenses) 3D kesiti.</em></figcaption>
  </div>
</figure>

- **Mikro-Katman Yapısı:** Taramalı Elektron Mikroskobu (SEM) incelemeleri, mikromercek tepesinden silisyum tabanına kadar olan toplam katman yüksekliğinin yalnızca $\approx 9.6\ \mu\text{m}$ olduğunu gösterir:
  1. *Üst Katman:* Kavisli organik mikromercek dizisi
  2. *Ara Katman:* Renk filtresi dizisi (RGB boyaları)
  3. *Taban Katmanı:* Fotodiyot kuyuları, yüzen difüzyon ve metal bağlantı yolları içeren silisyum taban.

<figure style="display:flex; justify-content: center; margin: 20px 0;">
  <div style="text-align: center;">
    <img src="../../../../../img/first-principles-of-computer-vision/overview-history-and-sensor-types-10.png" alt="Görüntü Sensörü SEM Enine Kesit Görüntüsü" style="display:flex; border-radius: 5px; justify-content: center; width: 500px;">
    <figcaption style="margin-top: 0.5em; text-align: center; font-size: 13px; color: #888;"><em>Görüntü Sensörü SEM Enine Kesit Görüntüsü: Taramalı Elektron Mikroskobu (SEM) ile çekilen; mikromercek, renk filtresi, metal yollar ve silikon tabakanın toplam 9.6 mikrometre kalınlığını gösteren gerçek nanoyapı görüntüsü.</em></figcaption>
  </div>
</figure>
