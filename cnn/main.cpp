#include <iostream>
#include <vector>
#include <string>
#include <random>    // Rastgele veri üretimi için
#include <numeric>   // std::iota için
#include <algorithm> // std::min, std::max için

// Kendi başlık dosyalarımızı dahil ediyoruz
#include "matrix.h"
#include "cnn_layers.h"
#include "cnn.h"

// Küçük ölçekli sahte görüntü verisi oluşturan fonksiyon
std::vector<Matrix> create_dummy_images(size_t num_samples, int img_rows, int img_cols) {
    std::vector<Matrix> images;
    images.reserve(num_samples); // Bellek ayırma

    std::random_device rd;
    std::mt19937 gen(rd());
    // 0.0 ile 1.0 arasında rastgele ondalık sayılar üret
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t k = 0; k < num_samples; ++k) {
        Matrix img(img_rows, img_cols);
        for (int i = 0; i < img_rows; ++i) {
            for (int j = 0; j < img_cols; ++j) {
                img(i, j) = dis(gen); // Her piksele rastgele değer ata
            }
        }
        images.push_back(img);
    }
    return images;
}

// Küçük ölçekli sahte etiket verisi (one-hot encoding) oluşturan fonksiyon
std::vector<Matrix> create_dummy_labels(size_t num_samples, size_t num_classes) {
    std::vector<Matrix> labels;
    labels.reserve(num_samples); // Bellek ayırma

    std::random_device rd;
    std::mt19937 gen(rd());
    // Sınıf sayısı aralığında rastgele tam sayılar üret (0'dan num_classes-1'e kadar)
    std::uniform_int_distribution<> dis(0, num_classes - 1);

    for (size_t k = 0; k < num_samples; ++k) {
        Matrix one_hot_label(num_classes, 1); // Tek bir sütun vektörü
        one_hot_label.zero(); // Tüm değerleri sıfırla
        one_hot_label(dis(gen), 0) = 1.0; // Rastgele bir sınıfı 1.0 yap
        labels.push_back(one_hot_label);
    }
    return labels;
}

int main() {
    // ---------------------- 1. Küçük Ölçekli Sahte Veri Oluşturma ----------------------
    const int IMG_ROWS = 10; // Görüntü yüksekliği (örneğin 10x10)
    const int IMG_COLS = 10; // Görüntü genişliği
    const size_t NUM_CLASSES = 3; // Sınıf sayısı (örneğin 3 farklı kategori)

    const size_t NUM_TRAIN_SAMPLES = 50; // Eğitim veri seti boyutu
    const size_t NUM_TEST_SAMPLES = 10;  // Test veri seti boyutu

    std::cout << "Creating dummy dataset..." << std::endl;
    std::vector<Matrix> train_images = create_dummy_images(NUM_TRAIN_SAMPLES, IMG_ROWS, IMG_COLS);
    std::vector<Matrix> train_labels = create_dummy_labels(NUM_TRAIN_SAMPLES, NUM_CLASSES);

    std::vector<Matrix> test_images = create_dummy_images(NUM_TEST_SAMPLES, IMG_ROWS, IMG_COLS);
    std::vector<Matrix> test_labels = create_dummy_labels(NUM_TEST_SAMPLES, NUM_CLASSES);

    std::cout << "Dummy data created: "
              << NUM_TRAIN_SAMPLES << " training samples, "
              << NUM_TEST_SAMPLES << " test samples." << std::endl;
    std::cout << std::endl;

    // ---------------------- 2. CNN Modelini Oluşturma ----------------------
    CNN model(0.01, "categorical_crossentropy"); // Öğrenme oranı biraz artırıldı

    // Basit bir CNN mimarisi
    // Giriş: 10x10 görüntü

    // İlk Evrişimsel Katman
    // Çıktı: (10 - 3)/1 + 1 = 8x8. 8 filtre ile => 8 adet 8x8 özellik haritası.
    // Matrix sınıfında bunlar (8 * 8) x 8 olarak birleştirilir.
    model.addConvolutionalLayer(8, 3); // 8 filtre, 3x3 kernel
    model.addActivationLayer("relu");  // ReLU aktivasyon

    // İlk Max Pooling Katmanı
    // Çıktı: (8 - 2)/2 + 1 = 4x4. Her özellik haritası için.
    // Matrix sınıfında bunlar (8 * 4) x 4 olarak birleştirilir.
    model.addMaxPoolingLayer(2); // 2x2 havuz, stride 2

    // Düzleştirme Katmanı
    // Önceki katmandan gelen 8x4x4 (32x4 birleşik Matrix) veriyi düzleştirir.
    // Toplam eleman: 8 * 4 * 4 = 128
    model.addFlattenLayer();

    // Tam Bağlantılı (Gizli) Katman
    // Girdi boyutu: 128 (flatten'dan gelen)
    model.addFullyConnectedLayer(128, 16); // 128 giriş, 16 nöron
    model.addActivationLayer("relu");      // ReLU aktivasyon

    // Çıkış Katmanı (Tam Bağlantılı)
    // Sınıflandırma için sınıf sayısı kadar çıkış.
    model.addFullyConnectedLayer(16, NUM_CLASSES); // 16 giriş, NUM_CLASSES çıkış
    model.addActivationLayer("softmax");           // Softmax aktivasyon

    // Model özetini göster
    model.summary();

    // ---------------------- 3. Modeli Eğitme ----------------------
    std::cout << "\nStarting model training with dummy data..." << std::endl;
    model.train(train_images, train_labels, {}, {}, // Doğrulama verisi sağlanmıyor
                /*epochs=*/20, /*batch_size=*/5, /*verbose=*/true); // Daha küçük epoch ve batch boyutu

    // ---------------------- 4. Modeli Değerlendirme ----------------------
    std::cout << "\nStarting model evaluation on dummy test data..." << std::endl;
    model.evaluate(test_images, test_labels, /*verbose=*/true);

    std::cout << "\nDummy CNN training and evaluation completed." << std::endl;

    return 0;
}
