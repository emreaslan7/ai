# Çoklu Özellikler (Multiple Features)

<!-- toc -->

## Giriş (Introduction)

Gerçek dünya senaryolarında, tek bir özellik (feature) genellikle doğru tahminler yapmak için yeterli değildir. Örneğin, bir evin fiyatını tahmin etmek istiyorsak, sadece büyüklüğünü (metrekare) kullanmak yeterli olmayabilir. Yatak odası sayısı, konum ve evin yaşı gibi diğer faktörler de önemli rol oynar.

Birden çok özelliğe sahip olduğumuzda, hipotez fonksiyonumuz (hypothesis function) şu şekilde genişler:

$$
h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
$$

burada:

- $ x_1, x_2, ..., x_n $ girdi özellikleridir (input features),
- $ \theta_0, \theta_1, ..., \theta_n $ öğrenmemiz gereken parametrelerdir (parameters) (ağırlıklar).

Örneğin, bir ev fiyatı tahmin modelinde hipotez fonksiyonu şöyle olabilir:

$$
h_{\theta}(x) = \theta_0 + \theta_1 (\text{Büyüklük}) + \theta_2 (\text{Yatak Odası Sayısı}) + \theta_3 (\text{Evin Yaşı})
$$

Bu, modelimizin birden çok faktörü dikkate almasını sağlayarak, tek bir özellik kullanmaya kıyasla doğruluğunu artırır.

---

## Vektörleştirme (Vectorization)

Hesaplamaları optimize etmek için hipotez fonksiyonumuzu matris gösterimi (matrix notation) ile temsil ederiz:

burada:

$ X $, eğitim örneklerini (training examples) içeren matristir

$ \theta $, parametre vektörüdür (parameter vector)

Bu, tek tek eğitim örnekleri üzerinde döngü yapmak yerine matris işlemlerini kullanarak verimli hesaplama yapmamızı sağlar.

### Neden Vektörleştirme?

Vektörleştirme, döngü kullanan işlemleri matris işlemlerine dönüştürme sürecidir. Bu, özellikle büyük veri kümeleriyle çalışırken hesaplama verimliliğini artırır. Bir döngü kullanarak tahminleri tek tek hesaplamak yerine, tüm hesaplamaları aynı anda gerçekleştirmek için doğrusal cebirden (linear algebra) yararlanırız.

Vektörleştirme olmadan (döngü kullanarak):

```python
m = len(X)  # Number of training examples
h = []
for i in range(m):
    prediction = theta_0 + theta_1 * X[i, 1] + theta_2 * X[i, 2] + ... + theta_n * X[i, n]
    h.append(prediction)
```

Vektörleştirme ile:

```python
h = np.dot(X, theta)  # Compute all predictions at once
```

Bu yöntem, matris işlemlerini verimli bir şekilde yürüten **NumPy** gibi optimize edilmiş sayısal kütüphanelerden yararlandığı için önemli ölçüde daha hızlıdır.

### Vektörleştirilmiş Maliyet Fonksiyonu (Vectorized Cost Function)

Benzer şekilde, çoklu özellikler için maliyet fonksiyonumuz (cost function) şöyledir:

$$ J(\theta) = \frac{1}{2m} \sum(h_\theta(x^{(i)}) - y^{(i)})^2 $$

Matrisler kullanılarak bu şu şekilde yazılabilir:

$$ J(\theta) = \frac{1}{2m} (X\theta - y)^T (X\theta - y) $$

Ve Python'da şu şekilde uygulanır:

```python
def compute_cost(X, y, theta):
    m = len(y)  # Number of training examples
    error = np.dot(X, theta) - y  # Compute (Xθ - y)
    cost = (1 / (2 * m)) * np.dot(error.T, error)  # Compute cost function
    return cost
```

Vektörleştirilmiş işlemler kullanarak, açık döngüler kullanmaya kıyasla önemli bir performans artışı elde ederiz.

---

## Özellik Ölçekleme (Feature Scaling)

Birden çok özellikle çalışırken, farklı özellikler arasındaki değer aralıkları önemli ölçüde değişebilir. Bu, gradyan inişinin (gradient descent) performansını olumsuz etkileyerek yavaş yakınsamaya veya verimsiz güncellemelere neden olabilir. **Özellik ölçekleme**, özellikleri benzer bir ölçeğe getirmek için normalize veya standardize etmekte kullanılan bir tekniktir ve gradyan inişinin verimliliğini artırır.

### Özellik Ölçekleme Neden Önemlidir?

- Büyük değerlere sahip özellikler maliyet fonksiyonuna hakim olabilir ve verimsiz güncellemelere yol açabilir.
- Özellikler benzer ölçekte olduğunda gradyan inişi daha hızlı yakınsar.
- Gradyanları hesaplarken sayısal kararsızlığı (numerical instability) önlemeye yardımcı olur.

### Özellik Ölçekleme Yöntemleri

#### 1. **Min-Maks Ölçekleme (Min-Max Scaling / Normalizasyon)**

Tüm özellik değerlerini sabit bir aralığa, tipik olarak 0 ile 1 arasına getirir:

$$x^{(i)}_{scaled} = \frac{x^{(i)} - x_{min}}{x_{max} - x_{min}}$$

- Veri dağılımının Gaussian (normal) olmadığı durumlar için en iyisidir.
- Aykırı değerlere (outliers) karşı hassastır, çünkü uç değerler aralığı etkiler.

#### 2. **Standardizasyon (Z-Score Normalizasyonu)**

Veriyi sıfır etrafında birim varyans ile merkezler:

$$x^{(i)}_{scaled} = \frac{x^{(i)} - \mu}{\sigma}$$

burada:

- $ \mu $ özellik değerlerinin ortalamasıdır (mean)
- $ \sigma $ standart sapmadır (standard deviation)

- Özellikler normal dağılım izlediğinde iyi çalışır.
- Aykırı değerlere karşı min-maks ölçeklemeye kıyasla daha az hassastır.

### Örnek

İki özelliğe sahip bir veri kümesi düşünelim: **Ev Büyüklüğü (m²)** ve **Yatak Odası Sayısı**.

| Ev Büyüklüğü (m²) | Yatak Odası |
| ----------------- | ----------- |
| 2100              | 3           |
| 1600              | 2           |
| 2500              | 4           |
| 1800              | 3           |

Min-maks ölçekleme kullanarak:

| Ev Büyüklüğü (ölçeklenmiş) | Yatak Odası (ölçeklenmiş) |
| -------------------------- | ------------------------- |
| 0,714                      | 0,5                       |
| 0,0                        | 0,0                       |
| 1,0                        | 1,0                       |
| 0,286                      | 0,5                       |

### Gradyan İnişinde Özellik Ölçekleme

Ölçekleme sonrasında, gradyan inişi güncellemeleri farklı özellikler arasında daha dengeli olacak ve daha hızlı ve daha kararlı bir yakınsama sağlayacaktır. Özellik ölçekleme, gradyan inişi gibi optimizasyon algoritmalarını içeren makine öğrenimi modellerinde kritik bir ön işleme adımıdır.

<br/>
<br/>
