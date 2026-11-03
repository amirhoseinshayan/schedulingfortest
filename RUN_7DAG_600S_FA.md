# گزارش و راهنمای نسخه هفت DAG متوالی در یک Episode

## هدف و دامنه تغییر

این نسخه از روی `MuxGNN_16Vehicles_4RSU_InstantTransfer_FullMigration.zip` ساخته شده است. هدف، رفع محدودیت آموزش روی یک DAG ثابت با **هفت DAG متفاوت در هر Episode** و یک بازه **۶۰۰ ثانیه‌ای پیوسته** از حرکت SUMO است. قرارداد قبلی **SUMO Trace آفلاین + Scheduler رویدادمحور + MuxGNN آفلاین + PPO آنلاین** حفظ شده است. فقط تولید DAG، اتصال آن به گراف/Simulator/Pretraining/PPO/HEFT، کنترل انتهای Episode و ثبت آمار مرتبط تغییر کرده است.

**بدون تغییر:** ۱۶ خودرو، ۴ سرور، پوشش V2V=۱۵۰ متر، V2I=۳۰۰ متر، گام SUMO=۰٫۰۵ ثانیه، حرکت و Rerouter، گرافیک SUMO، مدل ۵ رابطه‌ای و ۲ لایه MuxGNN، پارامترهای PPO، مدل A7/A12، سقف مهاجرت ۳ بار، فرض انتقال آنی داده، الگوریتم Reward و داده‌های اندازه‌گیری اصلی v4. اصلاح گرافیکی دایره پوشش سرورها نیز عمداً انجام نشده است.

## ۱. تولید DAGهای جدید

- هر Episode هفت **ساختار متفاوت** و ازپیش‌تعیین‌شده از نظر Seed دارد؛ Episode جدید هفت DAG تازه می‌گیرد.
- هر DAG **۶۰ تسک، ۱۲۶ یال وابستگی و ۸ نسل توپولوژیک** دارد. تعداد ریشه‌ها، آرایش لایه‌ها و والدهای تسک‌ها متفاوت‌اند؛ در نتیجه تعداد و ترتیب تسک‌های Ready متفاوت می‌شود.
- همه گراف‌ها Directed Acyclic، weakly connected و دارای مسیرهای معتبر هستند. fingerprint برای شناسایی گراف تکراری ثبت می‌شود.
- Runtime، Energy، Power و Gang Width هر تسک **دقیقاً از پروفایل مشتق‌شده پنج‌برابری v4** استفاده می‌کنند. هیچ زمان/انرژی اندازه‌گیری‌شده تازه‌ای اختراع نشده است. تنوع این نسخه در ساختار وابستگی‌هاست، نه در توزیع زمان‌های اجرای benchmark.
- Seedها در Train/Validation/Test در سه بازه جدا ساخته می‌شوند؛ تداخل DAG بین این مجموعه‌ها مجاز نیست. تولیدگر deterministic است.
- `dag_definitions_manifest.json` شامل **۱۶۸ گراف** برای ۲۴ سناریو است: ۱۲۶ تعریف گراف Train، ۲۱ Validation و ۲۱ Test. این‌ها **تعریف DAG** هستند، نه Snapshot حرکتی واقعی یا نتیجه Training.

## ۲. Critical Path و Deadline مستقل

برای DAG شماره i و هر تسک v:

`fastest_runtime(v) = min(runtime_A7(v), runtime_A12(v))`

`EF(v) = fastest_runtime(v) + max(EF(parent))`؛ برای ریشه‌ها max=0.

`CP_i = max(EF(v))` و `workflow_deadline_i = 1.15 * CP_i`.

`task_deadline(v) = 1.15 * EF(v)` **نسبت به زمان Release همان DAG** است. ضریب پنج‌برابری روی Runtime از قبل اعمال شده و دوباره اعمال نمی‌شود. CP و Deadline به تصمیم PPO یا HEFT وابسته نیستند.

## ۳. زمان، Release و Reset

- یک Episode تا **۷ DAG** دارد و طول قابل‌استفاده آن از اولین Sample معتبر با حضور همه ۱۶ خودرو، دقیقاً **۶۰۰ ثانیه** است.
- DAG ۱ در ابتدای این بازه آزاد می‌شود. DAG جدید **فقط بعد از تکمیل و تحویل تمام ۶۰ خروجی DAG قبلی**، در همان timestamp جهانی SUMO آزاد می‌شود.
- `start_sec` برای DAG بعدی برابر همان `state.physical_time_sec` پایان DAG قبلی است؛ موقعیت، اتصال‌ها و زمان جهانی SUMO هرگز به نقطه شروع بازنمی‌گردند.
- شمارنده نسبی زمان تسک‌های هر DAG از صفر شروع می‌شود تا Deadlineهای همان DAG درست سنجیده شوند؛ این موضوع به معنی Reset فیزیکی SUMO نیست.
- ساختار شبیه‌ساز جدید برای هر DAG باعث می‌شود Event Queue، Checkpoint، Migration pending و بافرهای قدیمی هیچ‌وقت به DAG بعدی نشت نکنند. قبل از Release بعدی، کنترل می‌شود که **همه هسته‌های رزروشده آزاد** و **هیچ خروجی بافری باقی نمانده باشد**.
- ظرفیت پس‌زمینه A7/A12 یک‌بار در ابتدای Episode نمونه‌گیری می‌شود و در هر هفت DAG **همان ظرفیت** حفظ می‌شود؛ در مرز DAG ظرفیت‌ها مجدداً تصادفی‌سازی نمی‌شوند.
- اگر هفت DAG قبل از ۶۰۰ ثانیه تمام شوند، Episode با `terminated=True` پایان طبیعی می‌یابد؛ نیازی به پرکردن مصنوعی بقیه زمان نیست.
- اگر زمان ۶۰۰ ثانیه برسد، DAG نیمه‌تمام و انرژی مصرف‌شده تا آن زمان صادقانه ثبت می‌شود؛ پایان زمان `truncated=True` است و به‌عنوان موفقیت گزارش نمی‌شود. DAGهای آزاد‌نشده نیز موفق محسوب نمی‌شوند. ۷ DAG **هدف ثابت** است، نه ادعای تضمین تکمیل ۷ DAG.

## ۴. انرژی، Reward و PPO

- انرژی هر DAG مستقل از صفر محاسبه می‌شود؛ مصرف کل Episode برابر جمع انرژی DAGهای تکمیل‌شده به‌علاوه مصرف ثبت‌شده DAG نیمه‌تمام است.
- انرژی یک معیار بهینه‌سازی/گزارش است؛ باتری فیزیکی و شارژدهی میان DAGها اضافه نشده‌اند.
- فرمول Reward قبلی **QoS × exp(-clip(E_task/E_ref_max,0,1))**، ضریب‌ها و پرداخت یک‌باره Reward **هنگام تحویل خروجی تسک** حفظ شده‌اند.
- هر هفت DAG **یک Episode واحد برای PPO** هستند. در مرز DAG، `terminated`/`truncated` به‌شکل مصنوعی فعال نمی‌شود. وزن‌های Policy، Optimizer و فرایند یادگیری از ابتدا شروع نمی‌شوند.
- محاسبه Advantage/Return مرز DAG را پایان Episode فرض نمی‌کند. PPO همچنان On-Policy و آنلاین در محیط Event-Driven است و از آینده Trace اطلاع نمی‌گیرد.
- کانال **موجود** زمان در Observation اکنون کسری از ۶۰۰ ثانیه سپری‌شده را نشان می‌دهد؛ هیچ بُعد گراف، Relation یا ساختار شبکه تغییر نکرده است.
- در ارزیابی، علاوه بر Reward مجموع، Reward میانگین هر تسک تکمیل‌شده، انرژی DAG و Episode، تعداد DAG تکمیل‌شده/ناتمام و Makespan ثبت شود.

## ۵. Pretraining آفلاین MuxGNN و تفکیک داده

- در `generate` برای **هر DAG واقعی سناریو**، Snapshot رویدادی از گراف همان DAG و موقعیت واقعی SUMO در لحظه تصمیم ساخته می‌شود. مسیر قدیمیِ خواندن همیشگی DAG v4 از Builder اصلاح شده و اکنون گراف با `workload` فعلی ساخته می‌شود.
- `loader` به‌جای ساختار یک‌DAG، Snapshotهای هفت DAG هر سناریو را recursive می‌خواند.
- `pretrain` همان objective و همان تعداد لایه/Relation نسخه قبلی را اجرا می‌کند، اما داده آموزش/اعتبارسنجی شامل DAGهای متنوع است.
- PPO Encoder آموزش‌دیده آفلاین را بارگذاری می‌کند؛ در Training خود PPO، Encoder طبق انتخاب قبلی frozen است.
- Seedهای مجموعه Test با Train و Validation مشترک نیستند. **برای Test واقعی، هم Snapshotهای ارزیابی باید تولید شوند و هم اجرای PPO آموزش‌دیده و HEFT روی DAGهای Test بررسی شود.**

## ۶. مقایسه منصفانه با HEFT

برای هر `episode_index` و `split=test`، هفت DAG و Seedهای آن، Trace حرکت، تعریف سخت‌افزار و ظرفیت اولیه یکسان‌اند. در HEFT نیز هر DAG در همان لحظه تکمیل DAG قبلی همان الگوریتم آزاد می‌شود؛ بنابراین **زمان دقیق Release می‌تواند به‌دلیل سرعت متفاوت سیاست‌ها فرق کند** و این یک پیامد طبیعی اجرای متوالی است. نسخه HEFT موجود، *online rank-based HEFT-style* است، نه HEFT کلاسیک آگاه از آینده. هر دو روش باید تعداد DAGهای تکمیل‌شده، تأخیر، انرژی و مقدار Reward را جدا گزارش کنند؛ در صورت ناتمام‌ماندن یکی، صرفاً میانگین DAGهای تکمیل‌شده معیار کافی نیست.

## ۷. دستورهای اجرا در Windows PowerShell

در پوشه `scheduling_utils`:

```powershell
$env:MUX_MOBILITY3 = "1"

# کنترل DAGها و گذار بین هفت Workflow (ابتدا fixture مصنوعی؛ بدون نیاز به SUMO)
python -m muxgnn.mobility3.verify_multi_dag

# ضبط Trace واقعی: توجه به ورود با تأخیر V15 حدود ثانیه ۳
# ۶۰۵ ثانیه ضبط، دست‌کم ۶۰۰ ثانیه بازه معتبر پس از ورود تمام خودروها می‌دهد.
python -m muxgnn.mobility3.capture --seconds 605 --seed 42

# حتما validation.json باید valid=true و بدون arrival/teleport/missing باشد.
# آزمون منطق هفت DAG با Trace واقعی
python -m muxgnn.mobility3.verify_multi_dag --trace data/muxgnn/v4_mobility_16vehicle_4edge_x5_seed42_step005_7dag_v2/sumo_trace.csv

# تولید Snapshotهای هفت DAG در هر سناریو، split مستقل برای آموزش و ارزیابی
python -m muxgnn.mobility3.generate --scenarios 24

# Pretraining روی گراف‌های متنوع
python -m muxgnn.mobility3.pretrain --epochs 10

# PPO آنلاین، یک Episode = هفت DAG متوالی
python -m muxgnn.mobility3.train --timesteps 20000

# HEFT-style برای همان DAGها و Trace در split=Test
python -m muxgnn.mobility3.heft_online

# ارزیابی PPO مقابل HEFT روی DAGهای دیده‌نشده
python -m muxgnn.mobility3.evaluate --episodes 3
```

**هشدار:** فایل خروجی `generate` ممکن است Snapshotهای زیاد ایجاد کند. برای آزمون اولیه با `--scenarios 8` شروع کنید تا یک سناریوی Test و یک Validation هم وجود داشته باشد؛ برای آزمایش اصلی تعداد سناریوها و Seedها را بیشتر کنید. اگر Capture نامعتبر بود، Train را آغاز نکنید.

## ۸. آزمون‌های انجام‌شده و محدودیت‌ها

- تولید deterministic و بررسی **۴۲ DAG بدون تکرار** در سه Split، ۶۰ تسک، ۱۲۶ یال، ۸ لایه، اتصال‌پذیری، Critical Path/Deadline، ثابت‌ماندن پروفایل‌ها: موفق.
- آزمون پیوستگی هفت DAG در Simulator با Trace **مصنوعی ثابت برای کنترل منطق**: هفت DAG، مجموع ۴۲۰ تسک تکمیل‌شده، زمان پیوسته، منابع ثابت و انرژی تجمعی صحیح: موفق.
- آزمون توقف ۶۰۰ ثانیه در میان DAG: بدون موفق‌نامیدن تسک نیمه‌تمام و با حفظ انرژی مصرف‌شده: موفق.
- سازگاری ساختار Graph Snapshot و ماسک اولیه PPO با stand-in سبک برای کتابخانه‌های نصب‌نشده: موفق؛ **جایگزین آزمایش با کتابخانه‌های اصلی نیست**.
- **در این محیط SUMO واقعی، PyTorch Geometric، Gymnasium و Stable-Baselines3 نصب نبودند.** بنابراین اجرای واقعی SUMO، Train کامل Encoder و PPO، تست ۶۰۰ ثانیه‌ای و نتایج علمی PPO/HEFT هنوز آزمایش نشده‌اند. هیچ نتیجه مصنوعی به‌عنوان نتیجه پژوهشی واقعی ارائه نشده است.

## ۹. فایل‌های مرتبط

تغییرات این نسخه محدود به `muxgnn/mobility3/{settings,simulator,builder,env,generate,loader,heft_online}.py` و `muxgnn/rl/adapter.py` (فقط کانال زمان جهانی)، و افزوده‌شدن `muxgnn/mobility3/{multi_dag,episode,export_dags,verify_multi_dag,evaluate}.py`، همین راهنما و پوشه مجزای تعریف DAGها است. **فایل‌های SUMO، تصویرهای گرافیکی، Reference v4، مدل و فایل‌های Training قدیمی خارج از این محدوده دست‌نخورده باقی مانده‌اند.**
