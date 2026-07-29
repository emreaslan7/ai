# Bilgisayarlı Görmeye Giriş

<!-- toc -->

## 1. Bilgisayarlı Görü Nedir?

Bilgisayarlı görü (computer vision), yalnızca yapay zekanın bir alt kümesi değil; çok disiplinli, köklü bir mühendislik ve bilim girişimidir. Fiziksel dünya ile sembolik anlama arasında köprü kurar; optik, sinyal işleme, elektrik mühendisliği ve bilgisayar biliminden beslenir.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../img/first-principles-of-computer-vision/introduction-to-computer-vision-01.png" style="display:flex; justify-content: center; width: 600px;"alt="Vision pipeline: light source, scene, camera, and Vision Software generating scene description"/>
</div>
<p style="text-align: center; font-size: 14px; color: #888; margin-top: -10px;"><i>Görü hattı: ışık kaynağı → sahne → kamera → Vision Software, sahne açıklamasını üretir</i></p>

Temel zorluk, ham sayısal dizileri—piksel verisini—anlamlı bir 3B ortam tanımına dönüştürmektir.

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 10px; gap: 20px; flex-wrap: wrap;">
    <div style="text-align: center;">
        <img src="../../img/first-principles-of-computer-vision/introduction-to-computer-vision-02.png" style="width: 300px;"alt="Black-and-white photo of two children showering — raw visual input"/>
        <p style="font-size: 13px; color: #888; margin-top: 4px;"><i>Ham görsel girdi: duş alan iki çocuk</i></p>
    </div>
    <div style="text-align: center;">
        <img src="../../img/first-principles-of-computer-vision/introduction-to-computer-vision-03.png" style="width: 320px;"alt="Numerical pixel matrix representation of the same photo"/>
        <p style="font-size: 13px; color: #888; margin-top: 4px;"><i>Aynı sahnenin sayısal piksel matrisi temsili</i></p>
    </div>
</div>

Bu alanda, misyonun tanımı çoğu zaman yöntemi belirler. Aşağıda bilgisayarlı görü araştırmasının üç temel felsefi dayanağının bir karşılaştırması yer almaktadır:

| Perspektif | Savunucu | Temel Felsefe |
|-------------|-----------|----------------|
| Görü olarak Taklit | David Marr | Biyolojik sistemlerin karmaşıklığını kopyalamak için insan görsel süreçlerini otomatikleştirmeyi amaçlar |
| Görü olarak Bilgi İşleme | Berthold Horn | Görüntü oluşumunu "tersine çevirme" işi olarak tanımlar — 2B projeksiyondan 3B gerçekliğe matematiksel olarak geri yürümek |
| Görü olarak İşlevsel Araç | Takeo Kanade | Görünün "eğlenceli" ama daha da önemlisi "kullanışlı" olduğunu vurgular; saf araştırma ile pratik uygulama arasında köprü kurar |

### 1.1 "İlk Prensipler" Felsefesi

Çağdaş derin öğrenme güçlü araçlar sunarken, **İlk Prensipler** yaklaşımı—matematiksel ve fiziksel temellere odaklanmak—genelleştirilebilir ve açıklanabilir yapay zeka için bir ön koşuldur. "Kara kutu" modellere güvenmek, gerçek yenilik için gereken yapısal anlayışı atlar.

> **Neden İlk Prensipler?** Fiziksel olaylar çoğu zaman zarif matematikle tanımlanabilir; bu da devasa veri kümelerini ve kapsamlı eğitim döngülerini gereksiz kılar.

Bu temelleri dört nedenle önceliyoruz:

1. **Kesinlik ve Özlülük** — Fiziksel olaylar genellikle zarif matematikle tanımlanabilir, büyük veri kümelerini gereksiz kılar.
2. **Hata Ayıklama ve Teşhis** — Bir görü sistemi başarısız olduğunda, ilk prensipler hatanın nedenini teşhis etmek için tek titiz çerçeveyi sağlar.
3. **Sentetik Veri Üretimi** — Gerçek dünya verisi toplamak pratik olmadığında veya tehlikeli olduğunda, matematiksel modeller yüksek kaliteli eğitim verisi üretmemizi sağlar.
4. **Bilimsel Merak** — Görsel olayların ardındaki "neden"i anlama içgüdüsü, yalnızca veri odaklı yöntemlerin gözden kaçırabileceği atılımlara yol açar.

---

## 2. İnsan Görme Sistemi: Biyoloji, Yanılma ve Belirsizlik

İnsan gözünü incelemek, yapay görü tasarlamak için gerekli bir başlangıç noktasıdır. Makineler ve insanlar çoğu zaman farklı hedeflere sahip olsa da—niteliksel navigasyon niceliksel ölçüme karşı—göz, verimli bilgi azaltımı için bir yol haritası sağlar.

### 2.1 Biyolojik Olarak Işının Takip Ettiği Patika

İnsan görsel sistemi, hızlı analiz için tasarlanmış karmaşık bir hiyerarşidir:

```mermaid
flowchart LR
    A["👁️ Göz ve Lens<br/><i>Birincil optik aşama</i>"] --> B["🧬 Retina<br/><i>Erken işleme + veri azaltma</i>"]
    B --> C["🔌 Optik Sinir<br/><i>Yüksek hızlı kanal</i>"]
    C --> D["🧠 LGN<br/><i>Röle istasyonu, bölgelere yönlendirir</i>"]
    D --> E["🎯 Görsel Korteks<br/><i>Şekil, renk, hareket, doku</i>"]
    
    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#e94560,color:#fff
    style C fill:#0f3460,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#16213e,stroke:#e94560,color:#fff
```

<div style="text-align: center;display:flex; justify-content: center; margin-bottom: 20px;">
    <img src="../../img/first-principles-of-computer-vision/introduction-to-computer-vision-04.png" style="display:flex; justify-content: center; width: 100%; max-width: 900px;"alt="Detailed brain anatomy: eye signals through LGN to visual cortex (V1, V2, MT/V5, V8)"/>
</div>
<p style="text-align: center; font-size: 14px; color: #888; margin-top: -10px;"><i>Biyolojik görme yolağı: retina → LGN → görsel korteks (V1, V2, MT/V5, V8)</i></p>

### 2.2 Niteliksel ve Niceliksel Görü

Mühendisler olarak şunu kabul etmeliyiz ki **insan görüşü niteliksel bir sistemdir**; oysa fabrika otomasyonu, tıbbi görüntüleme ve robotik niceliksel hassasiyet gerektirir. Bir insan bir yüzü anında tanıyabilir ancak bir bileşenin uzunluğunu milimetre hassasiyetiyle ölçemez. Aşırı güvenilirlik gerektiren görevler için insan biyolojisini taklit etmek çoğu zaman "yanlış" hedeftir; makineler biyolojik sistemlerin eksik olduğu ölçülebilir doğruluğu sağlamalıdır.

### 2.3 Görsel Yanılsamalar (İllüzyonlar)

İnsan görüşü göründüğünden daha yanılabilirdir; belirsizliği çözmek için genellikle içsel varsayımlara güvenir.

<br/>

**Örnek — Dongary Dalgası İllüzyonu:** Aşağıdaki statik yaprak deseni, gözün istemsiz mikro-sakadları nedeniyle titreşiyor veya hareket ediyormuş gibi görünür.

<div style="text-align: center;margin-bottom: 20px;">
    <img src="../../img/first-principles-of-computer-vision/introduction-to-computer-vision-05.png" style="width: 400px;"alt="Leaf illusion — static leaves appearing to move due to involuntary eye movements"/>
    <p style="font-size: 13px; color: #888; margin-top: 4px;"><i>Hareket algısı yaratan statik yaprak deseni — Dongary Dalgası illüzyonu</i></p>
</div>

| İllüzyon | Gösterdiği |
|----------|------------|
| **Fraser'in Spirali** | İç içe dairelerin beyin tarafından spiral olarak yorumlanması |
| **Adelson'un Satranç Gölgesi** | Beynin aydınlatmayı telafi etmesi — iki özdeş gri kare farklı görünür |
| **Dongary Dalgası** | İstemsiz göz hareketleri nedeniyle statik bir görüntüden hareket algılanması |
| **Ames Odası** | Perspektif ve göreceli boyut, insanların büyüyüp küçülüyormuş gibi göründüğü bir illüzyon yaratır |
| **Necker Küpü / Yüzler vs. Vazo** | Tek bir 2B görüntü birden fazla 3B veya sembolik yoruma izin verir |
| **Krater İllüzyonu** | "Yukarıdan aydınlatma" varsayımı — bir tümseği ters çevirmek onu krater gibi gösterir |
| **Kanizsa Üçgeni** | Beyin, pikselleri işlemekten (görmek) öte, veriyi "doldurur" (düşünür) — fiziksel olarak var olmayan bir üçgen algılar |

> **Önemli Çıkarım:** İnsanlar görsel deneyimleri aracılığıyla *düşünürken*, makineler önce radyometri ve geometrinin titiz merceğinden *hesaplamayı* öğrenmelidir.

---

## 3. Kapsanan Konular: Yol Haritası

Bu specializasyon, piksellerden algıya kadar tüm hattı kapsar ve altı modüle ayrılmıştır:

```mermaid
flowchart TB
    subgraph Foundations["🟦 Temeller"]
        direction TB
        A["Giriş<br/><i>CV nedir, insan görüşü</i>"]
        B["Görüntüleme<br/><i>Oluşum, sensörler, işleme</i>"]
    end
    
    subgraph Features["🟧 Öznitelikler & 2B"]
        C["Öznitelikler<br/><i>Kenarlar, SIFT, dikiş, yüzler</i>"]
    end
    
    subgraph Reconstruction["🟩 3B Yeniden Yapılandırma"]
        D["Yeniden Yapılandırma I<br/><i>Radyometri, fotometrik stereo</i>"]
        E["Yeniden Yapılandırma II<br/><i>Stereo, optik akış, SfM</i>"]
    end
    
    subgraph Perception["🟥 Algı"]
        F["Algı<br/><i>Takip, bölütleme, NN</i>"]
    end
    
    A --> B --> C --> D --> E --> F
    
    style A fill:#1a1a2e,stroke:#4cc9f0,color:#fff
    style B fill:#1a1a2e,stroke:#4cc9f0,color:#fff
    style C fill:#1a1a2e,stroke:#f72585,color:#fff
    style D fill:#1a1a2e,stroke:#06d6a0,color:#fff
    style E fill:#1a1a2e,stroke:#06d6a0,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style Foundations fill:transparent,stroke:#4cc9f0,color:#4cc9f0
    style Features fill:transparent,stroke:#f72585,color:#f72585
    style Reconstruction fill:transparent,stroke:#06d6a0,color:#06d6a0
    style Perception fill:transparent,stroke:#e94560,color:#e94560
```

### Modül Detayları

| # | Modül | Odak |
|---|-------|------|
| 1 | **Görüntüleme** | Görüntü oluşumu, sensörler, ikili görüntüler, görüntü işleme (konvolüsyon, Fourier) |
| 2 | **Öznitelikler** | Kenar/sınır tespiti, SIFT, görüntü dikişi, yüz tespiti |
| 3 | **Yeniden Yapılandırma I** | Radyometri, fotometrik stereo, gölgelemeden şekil, odak dışından derinlik |
| 4 | **Yeniden Yapılandırma II** | Kamera kalibrasyonu, stereo, optik akış, hareketten yapı |
| 5 | **Algı** | Nesne takibi, bölütleme, görünüm eşleme, sinir ağları |

---

## 4. Küresel Uygulamalar: Modern Dünyada Bilgisayarlı Görü

Görü, bir laboratuvar merakından, çeşitli sektörlerde gelişen küresel bir endüstriye dönüşmüştür.

<br/>

| Alan | Uygulamalar |
|------|-------------|
| **Endüstriyel / Verimlilik** | Fabrika otomasyonu, yüksek hızlı görsel denetim, plaka ve posta tarama için OCR |
| **Güvenlik / Kimlik** | DNA kadar benzersiz iris desenleriyle biyometri, güçlü yüz tanıma |
| **Tüketici Teknolojisi** | Optik fareler (mini görü sistemleri), oyun (Kinect/PlayStation), AR (Snapchat 3B filtreler) |
| **Akıllı Pazarlama** | Müşteri demografisini (yaş/cinsiyet) algılayıp hedefli ürün gösteren Shinagawa İstasyonu'ndaki otomatlar |
| **Görsel Arama** | Mobil cihazlarla anıtların ve nesnelerin anında tanımlanması |
| **İleri Mobilite** | Sensör füzyonu kullanan sürücüsüz arabalar, Mars Keşif Aracı'nın yabancı ortamlarda arazi haritalaması |
| **Yaratıcı / Tıbbi** | Sinema için hareket yakalama, X-ray, MR ve ultrason ile tıbbi teşhis |

---
