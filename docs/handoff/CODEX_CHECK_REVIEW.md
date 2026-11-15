# Codex: Check v2'yi incele (2026-11-14)

Claude, Check v2'yi kurdu. Tasarım, sonuçlar ve açık noktalar: `docs/CHECK_V2.md`. Kod: `lcr/calibrate.py`,
`scripts/20_check.py`–`23_owl_queries.py`, `lcr/pipeline.py: finalise`, `lcr/evidence.py`, `tests/test_calibrate.py`.

İnceleme kapsamı serbest. Tasarımı, kodu, sonuçları ve yorumları istediğin derinlikte sorgula. Gerekirse
yaklaşımı değiştir, başka yöntem dene, literatüre bak. Tek bağlayıcı şey AGENTS.md'deki genel kurallar.

Claude'un kendi şüphelerinin kısa listesi (kapsam sınırı değil, sadece not):
- λ seçimi B'nin katlar-dışı skorlarıyla yapıldı. `20_check.py` λ'yı hep 1.0 seçti.
- Captioner sinyallerinde, yeniden tokenleştirmeden sonra konum hizalaması.
- Karar noktası (kesinlik ≥ 0.5) Retell için doğru mu? Yoksa sürekli `p_correct` mi verilmeli?
- İkinci proxy (F30kE) referanslardan türüyor.
- Validation farkları için henüz güven aralığı yok.
- Sentetik bozmanın v2'de nasıl tanımlanacağı açık.
