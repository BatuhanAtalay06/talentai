# TalentAI

CV analizi ve iş ilanı eşleştirme için AI destekli bir Streamlit uygulaması.

## Hakkında

TalentAI, bir iş ilanını ve adayların CV'lerini (PDF/DOCX) Google Gemini ile
analiz ederek CV'leri yapılandırılmış veriye dönüştürür, embedding'ler
üretir ve cosine similarity ile iş ilanına en uygun adayları sıralar.
Tüm ilan ve aday kayıtları Postgres'te saklanır.

## Özellikler

- **Eşleştirme**: İş ilanı (pozisyon, tanım, nitelikler) girilir, embedding'i
  hesaplanır; yüklenen CV'ler Gemini ile ayrıştırılıp (ad, iletişim, deneyim,
  yetenekler, eğitim, özet) embedding'i çıkarılır ve iş ilanı vektörüyle
  cosine similarity üzerinden %eşleşme skoruna dönüştürülür (≥75 yüksek,
  ≥50 orta uyum).
- **Kayıtlı Kişiler**: Analiz edilmiş tüm adayların listesi (ad, e-posta,
  telefon, deneyim, yetenekler, eğitim, özet).
- **İş İlanları**: Kaydedilmiş tüm ilanların listesi.

## Teknoloji

- **UI**: Streamlit
- **AI**: Google Gemini (`google-genai`) — CV ayrıştırma (`gemini-2.5-flash`)
  ve embedding (`gemini-embedding-001`)
- **Veritabanı**: PostgreSQL (`psycopg2`) — `job_postings` ve `candidates`
  tabloları
- **Metin çıkarma**: `pypdf`, `python-docx`
- **Test**: `pytest`

## Kurulum

### 1. Ortam değişkenleri

Proje kök dizininde bir `.env` dosyası oluşturun:

```env
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/postgres
```

### 2a. Docker Compose ile çalıştırma (önerilen)

```bash
docker compose up -d --build
```

Bu komut hem Postgres'i (kalıcı volume ile) hem de uygulamayı ayağa
kaldırır. Uygulama `http://localhost:8501` adresinde çalışır.
`GEMINI_API_KEY` ve isteğe bağlı `POSTGRES_PASSWORD` `.env`'den okunur.

### 2b. Lokal (venv) ile çalıştırma

Postgres'in ayrıca ayakta olması gerekir (`docker compose up -d db` veya
kendi Postgres kurulumunuz):

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt   # Windows
venv/bin/pip install -r requirements.txt       # macOS/Linux

venv\Scripts\streamlit run app.py              # Windows
venv/bin/streamlit run app.py                  # macOS/Linux
```

## Test

```bash
pytest test_parser.py -v
```

Gemini çağrıları mock'lanmıştır; testler API anahtarı olmadan çalışır.
Veritabanı testleri (`TestDb`) çalışan bir Postgres'e ihtiyaç duyar ve
kendi kayıtlarını temizler.

## Lisans

MIT
