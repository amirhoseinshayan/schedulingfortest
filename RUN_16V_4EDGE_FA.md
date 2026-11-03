# اجرای نسخه ۱۶ خودرو و ۴ سرور (مشتق از v4)

این نسخه فقط توپولوژی منابع و نمایش لینک‌ها را توسعه می‌دهد. ۶۰ تسک، ۱۲۶ وابستگی، مقادیر مرجع A7/A12، Scaling پنج‌برابری زمان/Deadline/انرژی، توان مرجع، Reward، گام ۰٫۰۵ ثانیه، Seed=42، و حالت Trace آفلاین حفظ شده‌اند.

**توجه:** نام‌های قدیمی فایل‌ها و پوشه `sumo_three_edge` برای سازگاری دستورات حفظ شده‌اند، اما محتوای آن‌ها اکنون ۱۶ خودرو و ۴ سرور است.

## اجرای SUMO-GUI

در پوشه `scheduling_utils/sumo_three_edge` فایل `run_windows.bat` را اجرا کنید. برای نمایش **خطوط لینک V2V** به رنگ نارنجی و **خطوط V2I** به رنگ سبز باید با `run_gui_links.py` یا همین BAT اجرا کنید. بازکردن مستقیم `.sumocfg` خطوط متحرک را رسم نمی‌کند؛ فقط خودروها، سرورها و دایره‌های پوشش سرورها دیده می‌شوند.

فاصله V2V حداکثر **۱۵۰ متر** و V2I حداکثر **۳۰۰ متر** است؛ معیار هندسی فاصله اقلیدسی است و اثر موانع/رادیو واقعی مدل نشده است. فقط لینک‌های فعال نمایش داده می‌شوند. V0 مبدأ DAG است؛ مقصدهای Offloading شامل V1..V15 و EDGE0..EDGE3 با رعایت Action Mask هستند.

## ثبت Trace برای PPO آنلاین (روش توافق‌شده)

از پوشه `scheduling_utils` و با Python شامل SUMO TraCI:

```powershell
$env:MUX_MOBILITY3='1'
python sumo_three_edge/prepare_rerouters.py --seed 42 --seconds 600
python sumo_three_edge/validate_rerouters.py
python -m muxgnn.mobility3.validate_static
python -m muxgnn.mobility3.validate_scaling
python -m muxgnn.mobility3.capture --seconds 600 --seed 42
python -m muxgnn.mobility3.generate --scenarios 24
python -m muxgnn.mobility3.pretrain --epochs 10
python -m muxgnn.mobility3.train --timesteps 20000
python -m muxgnn.mobility3.heft_online
```

پیش از اولین Train واقعی `sumo_trace.validation.json` را بررسی کنید: `valid=true`، هیچ Teleport/Arrival، تمام V0..V15 و Link eventهای کافی. اجرای SUMO در این محیط سازنده ZIP انجام نشده است.

## قرارداد منصفانه منابع

هر خودروی محاسباتی: ۶ هسته A7 + ۶ هسته A12. هر Edge: ۱۲+۱۲ هسته. پس ظرفیت کل با نسخه ۶خودرو/۳سرور **بیشتر** شده است. نتایج دو توپولوژی را به‌عنوان دو سناریوی متفاوت گزارش دهید؛ PPO و HEFT در سناریوی جدید با همان Trace، محدودیت‌های لینک، منابع و workload مقایسه شوند.

Graph واقعی: ۶۰ task + ۱۶ vehicle + ۴ edge = **۸۰ node**؛ ۳ node types و ۵ relations و همان تعداد لایه‌های MuxGNN. ۴ سرور، شیء ثابت کنار جاده و رابط رادیویی فرضی دارند. زمان/انرژی شبکه هنوز مدل نشده‌اند (فقط محدودیت فاصله/اتصال). PPO این نسخه صرفاً Scheduling است؛ Migration قدیمی متصل نشده است.

## محدودیت اعتبارسنجی

آزمون Static، Graph-routing و Smoke Simulation با Trace ساختگی با اجرای SUMO واقعی متفاوت است. نگهداری ۱۶ خودرو، توزیع Link Up/Down و عملکرد Train نیازمند اجرای واقعی SUMO و محیط نصب‌شده `torch-geometric`/`stable-baselines3` هستند.

### به‌روزرسانی Migration (بدون تغییر داده‌ها یا SUMO)

در نسخه جدید، محیط Online PPO از Migration و تحویل آنی داده با شرایط اتصال پشتیبانی می‌کند. قبل از Training، راهنمای `RUN_MIGRATION_FA.md` را بخوانید. برای اجرای آزمون `python -m muxgnn.mobility3.verify_migration` و برای Train همان دستور `python -m muxgnn.mobility3.train` استفاده می‌شود. فرمول Reward حفظ شده ولی پاداش با انرژی واقعی قطعات اجراشده هنگام تحویل خروجی محاسبه می‌شود.
