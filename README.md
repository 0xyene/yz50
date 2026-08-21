# YZ50 — 1. Hafta

Sinir ağının temel parçalarının sıfırdan, kütüphane kullanmadan yazılması.

| Dosya | Görev |
|---|---|
| `yz1.py` | Tek nöron forward pass (ReLU) |
| `yz2.py` | Çok nöronlu katman, forward pass'in genişletilmesi |
| `yz3.py` | MSE loss fonksiyonu |
| `yz4.py` | Parametreyi tarayarak loss eğrisinin çizilmesi |
| `yz5.py` | Sayısal türev ile gradient descent döngüsü |

## Notlar

- `yz1`–`yz3` tamamen saf Python; hiçbir import yok.
- `yz4`–`yz5`'te numpy yalnızca değer aralığı üretmek, matplotlib yalnızca çizim için
  kullanıldı. İleri geçiş ve loss hesabı elle yazılmış hâliyle duruyor.
- `yz5` çıktısı `w = 1.9333`. Bu problemin en küçük kareler çözümü
  `Σx·y / Σx² = 58/30 = 1.9333` olduğu için gradient descent'in doğru çalıştığı
  bağımsız olarak doğrulanabiliyor.

Python 3.12 ile çalıştırıldı.
