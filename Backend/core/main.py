import random
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re


BOT_NAME = "دستیار خرید دیوار"


neighborhoods = [
    "سعادت آباد",
    "نیاوران",
    "شهرک غرب",
    "دزاشیب",
    "جردن",
    "زعفرانیه",
    "محمودیه",
    "اقدسیه",
    "تجریش",
    "تهرانپارس",
    "میرداماد",
    "شریعتی",
    "ونک",
    "سهروردی",
    "جمهوری",
    "انقلاب",
    "پاسداران",
    "دروازه دولت",
    "پیروزی",
    "میدان آزادی",
    "فردوسی",
    "محمدعلی‌جناح",
    "نواب",
    "نیاوران",
    "بهارستان",
    "شهرک اکباتان",
    "شهران",
    "توپخانه",
    "شهرک قدس",
    "پونک",
    "سهروردی",
    "پارک لاله",
    "دروازه شمیران",
    "جماران",
    "زرندیه",
    "ونک",
    "پالایشگاه",
    "نارمک",
    "تجریش",
    "جنت‌آباد",
    "پارک وی",
    "آفریقا",
    "سازمان آب",
    "شهرک غرب",
    "خیابان کارگر",
    "دماوند",
    "استادمعین",
    "زرندیه",
    "فرمانیه",
    "جوادیه",
    "دروازه شمیران",
    "پونک",
    "نیاوران",
    "جمشیدیه",
    "شهرک گلها",
    "قیطریه",
    "باغ فیض",
    "شهرک آزمایش",
    "الهیه",
    "دروس",
    "نیرو هوایی",
    "دولت آباد",
    "قزل قلعه",
    "قلهک",
    "زیباشهر",
    "ارژن",
    "شهرک ولیعصر",
    "خواجه نصیر",
    "تجریش",
    "اکباتان",
    "نازی آباد",
    "خیابان ستارخان",
    "سیته",
    "جنت آباد",
    "نیاوران",
    "محمود آباد",
    "شهرک آفریقا",
    "شهرک دانشگاه",
    "سازمان برنامه",
    "شهرک ستارخان",
    "جمهوری اسلامی",
    "بلوار کشاورز",
    "شهرک گلستان",
    "شهرک آپادانا",
    "شهرک صنعتی اشتهارد",
    "میدان آرژانتین",
    "شهرک قوام",
    "شهرک تکنولوژی پزشکی",
    "بهمنیار",
    "نارمک",
    "فرشته",
    "آجودانیه",
    "وحیدیه",
    "خزانه",
    "سهرورد",
    "درکه",
    "آزادشهر",
    "میدان توحید",
    "سیمین دشت",
    "میدان هفت تیر",
    "شهرک ژاندارمری",
    "میدان قنات کوثر",
    "شهرک سوهانک",
    "خیابان انقلاب",
    "خیابان میرعماد",
    "شهرک صنعتی پرند",
    "محمود آباد",
    "میدان شهدا",
    "شهرک سلمان فارسی",
    "نارمک",
    "نیرو هوایی",
    "جماران",
    "جردن",
    "استادمعین",
    "قیطریه",
    "اکباتان",
    "سعادت آباد",
    "شهرک غرب",
    "دزاشیب",
    "پونک",
    "زعفرانیه",
    "میرداماد",
    "شریعتی",
    "ونک",
    "سهروردی",
    "جمهوری",
    "انقلاب",
    "پاسداران",
    "دروازه دولت",
    "پیروزی",
    "میدان آزادی",
    "فردوسی",
    "محمدعلی‌جناح",
    "نواب",
    "نیاوران",
    "بهارستان",
    "شهرک اکباتان",
    "شهران",
    "توپخانه",
    "شهرک قدس",
    "پونک",
    "سهروردی",
    "پارک لاله",
    "دروازه شمیران",
    "جماران",
    "زرندیه",
    "ونک",
    "پالایشگاه",
    "نارمک",
    "تجریش",
    "جنت‌آباد",
    "پارک وی",
    "آفریقا",
    "سازمان آب",
    "شهرک غرب",
    "خیابان کارگر",
    "دماوند",
    "استادمعین",
    "زرندیه",
    "فرمانیه",
    "جوادیه",
    "دروازه شمیران",
    "پونک",
    "نیاوران",
    "جمشیدیه",
    "شهرک گلها",
    "قیطریه",
    "باغ فیض",
    "شهرک آزمایش",
    "الهیه",
    "دروس",
    "نیرو هوایی",
    "دولت آباد",
    "قزل قلعه",
    "قلهک",
    "زیباشهر",
    "ارژن",
    "شهرک ولیعصر",
    "خواجه نصیر",
    "تجریش",
    "اکباتان",
    "نازی آباد",
    "خیابان ستارخان",
    "سیته",
    "جنت آباد",
    "نیاوران",
    "محمود آباد",
    "شهرک نفت",
    "جمال زاده",
    "قلهک",
    "خیابان انقلاب",
    "پاسداران",
    "خیابان میرعماد",
    "میدان شهدا",
    "میدان توحید",
    "میدان هفت تیر",
    "شهرک ولیعصر",
    "شهرک آپادانا",
    "محمدعلی‌جناح",
    "شهرک آفریقا",
    "شهرک صنعتی اشتهارد",
    "زعفرانیه",
    "شریعتی",
    "تجریش",
    "میدان قنات کوثر",
    "خزانه",
    "سعادت آباد",
    "جمهوری اسلامی",
    "دزاشیب",
    "اکباتان",
    "سهروردی",
    "ونک",
    "جردن",
    "پونک",
    "توپخانه",
    "شهرک قدس",
    "میرداماد",
    "فردوسی",
    "شهرک گلها",
    "بهارستان",
    "نواب",
    "پیروزی",
    "باغ فیض",
    "دماوند",
    "نیاوران",
    "آفریقا",
    "نارمک",
    "زرندیه",
    "آجودانیه",
    "وحیدیه",
    "شهرک آزمایش",
    "سازمان برنامه",
    "شهرک دانشگاه",
    "شهرک ستارخان",
    "میدان آرژانتین",
    "خیابان کارگر",
    "شهرک گلستان",
    "شهرک صنعتی پرند",
    "شهرک قوام",
    "شهرک سوهانک",
    "بهمنیار",
    "دروازه دولت",
    "میدان آزادی",
    "جماران",
    "پونک",
    "نیاوران",
    "جمشیدیه",
    "شهرک گلها",
    "قیطریه",
    "باغ فیض",
    "شهرک آزمایش",
    "الهیه",
    "دروس",
    "نیرو هوایی",
    "دولت آباد",
    "قزل قلعه",
    "قلهک",
    "زیباشهر",
    "ارژن",
    "شهرک ولیعصر",
    "خواجه نصیر",
    "تجریش",
    "اکباتان",
    "نازی آباد",
    "خیابان ستارخان",
    "سیته",
    "جنت آباد",
    "نیاوران",
    "محمود آباد",
    "شهرک نفت",
    "جمال زاده",
    "قلهک",
    "خیابان انقلاب",
    "پاسداران",
    "خیابان میرعماد",
    "میدان شهدا",
    "میدان توحید",
    "میدان هفت تیر",
    "شهرک ولیعصر",
    "شهرک آپادانا",
    "محمدعلی‌جناح",
    "شهرک آفریقا",
    "شهرک صنعتی اشتهارد",
    "زعفرانیه",
    "شریعتی",
    "تجریش",
    "میدان قنات کوثر",
    "خزانه",
    "سعادت آباد",
    "جمهوری اسلامی",
    "دزاشیب",
    "اکباتان",
    "سهروردی",
    "ونک",
    "جردن",
    "پونک",
    "توپخانه",
    "شهرک قدس",
    "میرداماد",
    "فردوسی",
    "شهرک گلها",
    "بهارستان",
    "نواب",
    "پیروزی",
    "باغ فیض",
    "دماوند",
    "نیاوران",
    "آفریقا",
]

greeting = [
    "سلام! من میخوام کمکت کنم خونه‌ی مد نظرت رو پیدا کنی.",
    "سلام من قراره بهت کمک کنم به خونه‌ای که میخوای بخری راحت‌تر برسی.",
    "سلاام من دوست دارم کمکت کنم که خونه‌ی مورد نظرت رو سریع‌تر پیدا کنی.",
    "سلام! من اینجام تا بهت کمک کنم خونه‌ی مورد نظرت رو پیدا کنی.",
    "سلام! به خاطر تو، من اینجام تا بهت کمک کنم خونه‌ی دلخواهت رو پیدا کنی.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی مورد نظرت رو سریع‌تر پیدا کنی.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی رویاییت رو سریع‌تر پیدا کنی.",
    "سلام! اینجا هستم تا بهت کمک کنم خونه‌ی مد نظرت رو پیدا کنی.",
    "سلام! من اینجا هستم تا بهت کمک کنم به خونه‌ی انتخابیت برسی.",
    "سلام! به خاطر تو، من اینجا هستم تا بهت کمک کنم خونه‌ی مورد علاقه‌ات رو پیدا کنی.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی رویاهات رو پیدا کنی.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی ایده‌آلت رو به دست بیاری.",
    "سلام! به خاطر تو، من اینجا هستم تا بهت کمک کنم خونه‌ی مد نظرت رو بهتر و سریع‌تر پیدا کنی.",
    "سلام! من اینجا هستم تا بهت کمک کنم به خونه‌ی دلخواهت برسی.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی رویاییت رو به دست بیاری.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی ایده‌آلت رو پیدا کنی.",
    "سلام! به خاطر تو، من اینجا هستم تا بهت کمک کنم خونه‌ی مد نظرت رو پیدا کنی.",
    "سلام! من اینجا هستم تا بهت کمک کنم به خونه‌ی رویاهات برسی.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی ایده‌آلت رو پیدا کنی.",
    "سلام! به خاطر تو، من اینجا هستم تا بهت کمک کنم خونه‌ی مورد علاقه‌ات رو پیدا کنی.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی رویاییت رو به دست بیاری.",
    "سلام! من اینجا هستم تا بهت کمک کنم خونه‌ی ایده‌آلت رو بهتر و سریع‌تر پیدا کنی."
]
    

district = [
    "میخوای خونه‌ات توی چه محله‌ای باشه؟",
    "خونه‌ای که دنبالش می‌گردی میخوای کدوم سمت تهران باشه؟",
    "کدوم محله رو ترجیح می‌دی خونه‌ات باشه؟",
    "خونه‌ای که دنبالشی، دوست داری تو کدوم محدوده تهران باشه؟",
    "خونه‌تو تو کدوم محله می‌خوای داشته باشی؟",
    "می‌خوای خونه‌ت تو کدوم قسمت تهران باشه؟",
    "به دنبال یه خونه‌ای می‌گردی؟ خونه‌ت تو کدوم محله تهران باشه؟",
    "محله‌ای رو برای خونه‌ت انتخاب کن.",
    "دنبال خونه‌ای هستی؟ می‌خوای تو کدوم قسمت تهران باشه؟",
    "دنبال خونه‌ای هستی؟ خونه‌ت تو کدوم محله از تهران باشه؟",
    "می‌خوای خونه‌ت رو تو کدوم قسمت تهران بنا کنی؟",
    "می‌خوای خونه‌ت تو کدوم منطقه شهر تهران باشه؟",
    "محله‌ای که می‌خوای خونه داشته باشی رو انتخاب کن.",
    "می‌خوای خونه‌ت تو کدوم منطقه شهر تهران قرار بگیره؟",
    "برای خونه‌ت کدوم محله رو ترجیح می‌دی؟",
    "خونه‌ت رو تو کدوم قسمت تهران می‌خوای بنا کنی؟",
    "دنبال یه خونه‌ای هستی؟ می‌خوای تو کدوم منطقه تهران باشه؟",
    "می‌خوای خونه‌ت تو کدوم محله تهران قرار بگیره؟",
    "برای خونه‌ت یه محله رو انتخاب کن.",
    "خونه‌ت رو تو کدوم مکان از تهران می‌خوای؟",
    "دنبال خونه‌ای هستی؟ تو کدوم محله از تهران می‌گردی؟",
    "می‌خوای خونه‌ت تو کدوم قسمت تهران باشه؟"
]

cost = [
    "میخوای حدودا قیمتش چقدر باشه؟",
    "دوست داری توی چه رنج قیمتی باشه؟",
    "قیمت حدودی مورد نظرت چقدره؟",
    "دوست داری قیمتش توی کدوم بازه باشه؟",
    "می‌خوای حدوداً چقدر براش پول بدی؟",
    "قیمتی که تو دلت هست رو بفرما.",
    "دوست داری حدوداً چقدر براش هزینه کنی؟",
    "بهتره قیمتی که در نظر داری رو بدونم.",
    "قیمتی که بهش فکر می‌کنی، توی چه بازه‌ایه؟",
    "دوست داری قیمتش در حدود کدوم مقداری باشه؟",
    "می‌خوای قیمت حدودی این مورد رو بدونی؟",
    "قیمت تقریبی رو بفرما، خواهشاً.",
    "دوست داری در چه محدوده‌ای قیمت مشابه داشته باشه؟",
    "بهتره درباره قیمتش نظرت رو بدونم.",
    "قیمتی رو که تو ذهنت داری، بگو.",
    "دوست داری قیمتش توی چه رنجی باشه؟",
    "می‌خوای قیمت تقریبی این مورد رو بدونی؟",
    "قیمتی که بهش فکر می‌کنی، تو کدوم مقداره؟",
    "دوست داری قیمتش در چه بازه‌ای باشه؟",
    "می‌خوای حدوداً چقدر براش هزینه کنی؟",
    "قیمت تقریبی این مورد رو می‌خوای بدونی؟",
    "دوست داری در کدوم محدوده قیمتی مشابه داشته باشه؟"
]


dimensions = [
    "میخوای حدودا چند متر باشه؟",
    "ابعاد خونه‌ات میخوای چند متر باشه؟",
    "میخوای توی چه رنج متراژی باشه؟",
    "تقریباً چند متر مورد نظرته؟",
    "بهتره متراژ خونه‌ت رو بدونم، دوست داری چند متر باشه؟",
    "می‌خوای ابعاد خونه‌ت چند متری باشه؟",
    "توی چه رنجی دوست داری متراژ خونه‌ت باشه؟",
    "دوست داری حدوداً چند متر باشه؟",
    "می‌خوای متراژ خونه‌ت رو تقریباً چقدر بدونی؟",
    "متراژی که در نظر داری رو بگو.",
    "دوست داری ابعاد خونه‌ت تو چه بازه‌ای باشه؟",
    "می‌خوای ابعاد حدودی خونه‌ت رو بدونی؟",
    "بهتره متراژی که بهش فکر می‌کنی رو بفرما.",
    "دوست داری متراژ خونه‌ت توی چه محدوده‌ای باشه؟",
    "متراژ حدودی خونه‌ت رو بگو، خواهشاً.",
    "می‌خوای متراژ خونه‌ت رو در حدود کدام مقداری بدونی؟",
    "دوست داری متراژ خونه‌ت توی چه رنجی باشه؟",
    "تقریباً چند متر رو براش در نظر داری؟",
    "متراژ خونه‌ت رو تقریباً چقدر می‌خوای؟",
    "دوست داری ابعاد خونه‌ت در چه بازه‌ای باشه؟",
    "می‌خوای حدوداً چند متر برای خونه انتخاب کنی؟",
    "بهتره متراژی که در ذهنت هست رو بگو."
]


rooms = [
    "میخوای چند تا اتاق داشته باشه؟",
    "دوست داری حدودا چند تا اتاق داشته باشه؟",
    "میخوای خونه‌ات چند خوابه باشه؟",
    "دوست داری چند تا اتاق در خانه داشته باشی؟",
    "تعداد اتاق‌های خونه‌ت رو می‌خوای چند تا باشه؟",
    "می‌خوای توی خانه‌ت حدوداً چند اتاق داشته باشی؟",
    "اتاق‌های خونه‌ت رو چند تا می‌خوای؟",
    "دوست داری حدوداً چند تا اتاق در خانه داشته باشی؟",
    "تعداد اتاق‌هایی که دوست داری در خانه داشته باشی رو بگو.",
    "می‌خوای در خانه‌ت چند خوابه داشته باشی؟",
    "دوست داری خواب خونه‌ت چند خوابه باشه؟",
    "تعداد خواب‌های خانه‌ت رو می‌خوای چند تا باشه؟",
    "می‌خوای در خانه‌ت حدوداً چند خوابه داشته باشی؟",
    "دوست داری خواب خونه‌ت رو چند تا بدونی؟",
    "تعداد خواب‌هایی که بهش فکر می‌کنی رو بگو.",
    "می‌خوای خواب‌های خانه‌ت تو چه بازه‌ای باشه؟",
    "دوست داری تعداد اتاق خواب‌های خانه‌ت توی چه رنجی باشه؟",
    "تعداد خواب‌های خانه‌ت رو در حدود کدام مقداری بدونی؟",
    "می‌خوای تعداد اتاق‌ خواب‌های خانه‌ت رو تقریباً چقدر بدونی؟",
    "دوست داری تعداد اتاق‌ خواب‌های خانه‌ت در چه بازه‌ای باشه؟",
    "می‌خوای حدوداً چند تا اتاق خواب‌های در خانه داشته باشی؟",
    "بهتره تعداد خواب‌های که در ذهنت هست رو بگو."
]

floors = [
    "دوست داری خونه‌ات چند طبقه باشه؟",
    "میخوای چند طبقه باشه؟",
    "میخوای چند طبقه داشته باشه؟",
   "دوست داری خونه‌ات چند طبقه داشته باشه؟",
    "تعداد طبقه‌های خونه‌ات رو میخوای چند تا باشه؟",
    "میخوای توی خونه‌ت حدوداً چند طبقه داشته باشی؟",
    "طبقه‌های خونه‌ت رو چند تا میخوای؟",
    "دوست داری حدوداً چند طبقه در خانه داشته باشی؟",
    "تعداد طبقه‌هایی که دوست داری در خانه داشته باشی رو بگو.",
    "میخوای خونه‌ات چند طبقه باشه؟",
    "دوست داری خونه‌ات چند طبقه داشته باشه؟",
    "تعداد طبقه‌های خونه‌ات رو میخوای چند تا باشه؟",
    "میخوای توی خونه‌ت حدوداً چند طبقه داشته باشی؟",
    "دوست داری تعداد طبقه‌های خونه‌ت رو چند تا بدونی؟",
    "تعداد طبقه‌هایی که بهش فکر میکنی رو بگو.",
    "میخوای طبقه‌های خونه‌ت تو چه بازه‌ای باشه؟",
    "دوست داری تعداد طبقه‌های خونه‌ت توی چه رنجی باشه؟",
    "تعداد طبقه‌های خونه‌ت رو در حدود کدام مقداری بدونی؟",
    "میخوای تعداد طبقه‌های خونه‌ت رو تقریباً چقدر بدونی؟",
    "دوست داری تعداد طبقه‌های خونه‌ت در چه بازه‌ای باشه؟",
    "میخوای حدوداً چند طبقه در خانه داشته باشی؟",
    "بهتره تعداد طبقه‌هایی که در ذهنت هست رو بگو."
]

construction = [
    "چند سال ساخت باشه؟",
    "میخوای خونه‌ات چند سال ساخت باشه؟",
    "چند سال ساخت میخوای باشه خونه‌ات؟",
    "دوست داری خونه‌ات چند سال ساخت باشه؟",
    "تاریخ ساخت خونه‌ت رو چند سال میخوای باشه؟",
    "میخوای توی خونه‌ت حدوداً چند سال ساخت داشته باشی؟",
    "سن خونه‌ت رو چند سال میخوای؟",
    "دوست داری حدوداً چند سال ساخت در خانه داشته باشی؟",
    "تاریخ ساختی که دوست داری در خانه داشته باشی رو بگو.",
    "میخوای خونه‌ات چند سال ساخت باشه؟",
    "دوست داری خونه‌ات چند سال ساخت داشته باشه؟",
    "تاریخ ساخت خونه‌ات رو میخوای چند سال باشه؟",
    "میخوای توی خونه‌ت حدوداً چند سال ساخت داشته باشی؟",
    "دوست داری تاریخ ساخت خونه‌ت رو چند سال بدونی؟",
    "تاریخ ساختی که بهش فکر میکنی رو بگو.",
    "میخوای سن خونه‌ت تو چه بازه‌ای باشه؟",
    "دوست داری تاریخ ساخت خونه‌ت توی چه رنجی باشه؟",
    "تاریخ ساخت خونه‌ت رو در حدود کدام مقداری بدونی؟",
    "میخوای تاریخ ساخت خونه‌ت رو تقریباً چقدر بدونی؟",
    "دوست داری تاریخ ساخت خونه‌ت در چه بازه‌ای باشه؟",
    "میخوای حدوداً چند سال ساخت در خانه داشته باشی؟",
    "بهتره تاریخ ساختی که در ذهنت هست رو بگو."
    ]

facilities = [
    "چه امکاناتی میخوای خونه‌ات داشته باشه؟",
    "چه امکانات خاصی مد نظرته؟",
    "ویژگی خاصی هم مد نظرت هست؟",
    "دوست داری خونه‌ات چه امکاناتی داشته باشه؟",
    "تجهیزات خاصی که دوست داری در خانه باشه چیا هستن؟",
    "میخوای توی خونه‌ت چه امکاناتی رو داشته باشی؟",
    "امکانات خونه‌ت رو چه جوری میخوای؟",
    "دوست داری حدوداً چه امکانات خاصی در خانه داشته باشی؟",
    "ویژگی‌های خاصی که دوست داری در خانه داشته باشی رو بگو.",
    "میخوای خونه‌ات چه امکاناتی داشته باشه؟",
    "دوست داری خونه‌ات چه امکاناتی داشته باشه؟",
    "تجهیزات خاصی که دوست داری در خانه باشه چیا هستن؟",
    "میخوای توی خونه‌ت چه امکاناتی رو داشته باشی؟",
    "دوست داری تجهیزات خاصی در خانه داشته باشی؟",
    "امکانات خاصی که بهش فکر میکنی رو بگو.",
    "میخوای ویژگی‌های خونه‌ت تو چه بازه‌ای باشه؟",
    "دوست داری تجهیزات خاصی در خانه داشته باشی؟",
    "ویژگی‌های خاصی که دوست داری در خانه داشته باشی رو بگو.",
    "میخوای امکانات خونه‌ت رو تقریباً چقدر بدونی؟",
    "دوست داری تجهیزات خاصی در خانه داشته باشی؟",
    "امکانات خاصی که بهش فکر میکنی رو بگو.",
    "میخوای حدوداً چه امکاناتی در خانه داشته باشی؟"
]

district_question = "توی چه محله‌ای میخواد باشه؟"
cost_question = "قیمتش میخواد تقریبا چقدر باشه؟"
dimensions_question = "چند متر مد نظرشه؟"
rooms_question = "چند تا اتاق میخواد داشته؟"
floors_question = "چند طبقه میخواد باشه؟"
construction_question = "چند سال ساخت میخواد باشه؟"
facilities_question = "چه امکاناتی میخواد داشته باشه؟"


def run_model(paragraph, question, tokenizer, comprehension_model, **generator_args):
    input_ids = tokenizer.encode(question + "\n" + paragraph, return_tensors="pt")
    res = comprehension_model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print('output is', output)
    return output

def comprehension_on_message(message, question, comprehension_model, tokenizer):
    return run_model(
        message,
        question, 
        tokenizer,
        comprehension_model
    )


def shuffle_lst(lst):
    random.shuffle(lst)
    return 0


def find_k_most_relevant(query_similarity_vector, boostan_poems, k):
    results = []
    for i in range(k):
        max_value = max(query_similarity_vector)
        max_index = query_similarity_vector.index(max_value)
        results.append(boostan_poems[max_index])
        query_similarity_vector[max_index] = -1
    return results


def random_generator(current_questions, previous_questions_embeddings, model):
    future_question = random.choice(current_questions)
    previous_questions_embeddings.append(model.encode(future_question))
    return previous_questions_embeddings, future_question


def find_neighbour(res):
    for n in neighborhoods:
        if n in res:
            return n


RANDOMNESS_PARAM = 0.5
def embedding_generator(model, previous_questions_embeddings, current_questions):
    i = 1
    s = 0
    average_embedding = np.zeros(768, dtype='float32')
    print(len(previous_questions_embeddings))
    for q in previous_questions_embeddings: 
        # print(np.shape(previous_questions_embeddings[0]), np.shape(average_embedding))
        average_embedding += (previous_questions_embeddings[0] * i)
        s += i
        i += 1
    average_embedding /= s
    # print(average_embedding)
    current_questions_embeddings = []
    for q in current_questions:
        current_questions_embeddings.append(model.encode(q))
    
    if random.uniform(0, 1) <= RANDOMNESS_PARAM:
        future_question = random.choice(current_questions)
        previous_questions_embeddings.append(model.encode(future_question))
        return previous_questions_embeddings, future_question
    else:
        query_similarity_vector = cosine_similarity(
        [average_embedding],
        current_questions_embeddings
        )
        query_similarity_vector = list(query_similarity_vector[0])
        future_question = find_k_most_relevant(query_similarity_vector, current_questions, 1)[0]
        previous_questions_embeddings.append(model.encode(future_question))
        return previous_questions_embeddings, future_question


def convert_persian_number(number):
    persian_numbers = {
        'صفر': 0,
        'یک': 1,
        'یه': 1,
        'دو': 2,
        'سه': 3,
        'چهار': 4,
        'پنج': 5,
        'شش': 6,
        'هفت': 7,
        'هشت': 8,
        'نه': 9,
        'ده': 10,
        'یازده': 11,
        'دوازده': 12,
        'سیزده': 13,
        'چهارده': 14,
        'پانزده': 15,
        'شانزده': 16,
        'هفده': 17,
        'هجده': 18,
        'نوزده': 19,
        'بیست': 20,
        'سی': 30,
        'چهل': 40,
        'پنجاه': 50,
        'شصت': 60,
        'هفتاد': 70,
        'هشتاد': 80,
        'نود': 90,
        'صد': 100,
        'دویست': 200,
        'سیصد': 300,
        'چهارصد': 400,
        'پانصد': 500,
        'ششصد': 600,
        'هفتصد': 700,
        'هشتصد': 800,
        'نهصد': 900,
        'هزار': 1000,
        'میلیون': 1000000,
        'میلیارد': 1000000000,
        'میلیارد': 1000000000,
    }

    total = 0
    partial_sum = 0

    for word in number.split():
        if word in persian_numbers:
            value = persian_numbers[word]
            if value >= 1000:
                total += partial_sum * value
                partial_sum = 0
            else:
                partial_sum += value
        else:
            return None  # Invalid number

    return total + partial_sum


def extract_numbers_with_flags(sentence):
    numbers = []
    flags = []
    pattern = r'(?:تقریباً|اردر|نزدیک|حدود|زیر|کمتر از|بالای|بیشتر از|بین)\s+(\d+(?:\.\d+)?)\s*(?:تا|و|یا)\s*(\d+(?:\.\d+)?)|\b(\d+(?:\.\d+)?)\b'
    matches = re.findall(pattern, sentence)
    for match in matches:
        for i in range(len(match)):
            if match[i]:
                if i < 2:
                    start = float(match[i])
                    end = float(match[i+1])
                    numbers.extend(range(int(start), int(end) + 1))
                    if i == 0:
                        flags.extend(["Approximation"] * (int(end) - int(start) + 1))
                    elif i == 1:
                        flags.extend(["Between"] * (int(end) - int(start) + 1))
                else:
                    numbers.append(int(match[i]))
                    if match[i-1] == "زیر" or match[i-1] == "کمتر از":
                        flags.append("Less")
                    elif match[i-1] == "بالای" or match[i-1] == "بیشتر از":
                        flags.append("More")
                    else:
                        flags.append("Approximation")
                break
    return numbers, flags


def is_numeric_string(string):
    pattern = r'^(\d+)$|^(\d+)$'
    if re.match(pattern, string):
        return True
    return False

def convert_persian_number(number):
    persian_numbers = {
        'صفر': 0,
        'یک': 1,
        'یه': 1,
        'دو': 2,
        'سه': 3,
        'چهار': 4,
        'پنج': 5,
        'شش': 6,
        'هفت': 7,
        'هشت': 8,
        'نه': 9,
        'ده': 10,
        'یازده': 11,
        'دوازده': 12,
        'سیزده': 13,
        'چهارده': 14,
        'پانزده': 15,
        'شانزده': 16,
        'هفده': 17,
        'هجده': 18,
        'نوزده': 19,
        'بیست': 20,
        'سی': 30,
        'چهل': 40,
        'پنجاه': 50,
        'شصت': 60,
        'هفتاد': 70,
        'هشتاد': 80,
        'نود': 90,
        'صد': 100,
        'دویست': 200,
        'سیصد': 300,
        'چهارصد': 400,
        'پانصد': 500,
        'ششصد': 600,
        'هفتصد': 700,
        'هشتصد': 800,
        'نهصد': 900,
        'هزار': 1000,
        'میلیون': 1000000,
        'میلیارد': 1000000000,
        'میلیارد': 1000000000,
    }

    total = 0
    partial_sum = 0

    and_flag = 0
    converted_sentence = []
    for word in number.split():
        if word in persian_numbers:
            and_flag = 0
            value = persian_numbers[word]
            if value >= 1000:
                total += partial_sum * value
                partial_sum = 0
            else:
                partial_sum += value
        elif is_numeric_string(word):
            value = int(word)
            if value >= 1000:
                total += partial_sum * value
                partial_sum = 0
            else:
                partial_sum += value

        else:
            if word == "و" and and_flag != 1:
                and_flag = 1
                continue
            elif and_flag == 1:
                converted_sentence.append("و")
                and_flag = 2
            if total != 0 or partial_sum != 0:
                converted_sentence.append(str(total + partial_sum))
                total = 0
                partial_sum = 0
            print("yes", word)
            converted_sentence.append(word)

    if total != 0 or partial_sum != 0:
        converted_sentence.append(str(total + partial_sum))
        total = 0
        partial_sum = 0

    return ' '.join(converted_sentence)


def extract_facilities(sentence):
    facilities = []
    persian_facilities = [
        "آسانسور",
        "بالکن",
        "تراس",
        "پارکینگ",
        "انباری",
        "گاز",
        "آب",
        "برق",
        "شوفاژ",
        "کولر",
        "سرایدار",
        "استخر",
        "سونا",
        "جکوزی",
        "واحد آشپزی",
        "ماشین لباسشویی",
        "پنجره دو جداره",
        "کابینت MDF",
        "پارکت",
        "کف سرامیک",
        "پکیج",
        "بخاری",
        "کولر گازی",
        "سیستم گرمایشی",
        "کمد دیواری",
        "مبله",
        "بالکن گرمایشی",
        "کاغذ دیواری",
        "شومینه"
    ]

    for facility in persian_facilities:
        if facility in sentence:
            facilities.append(facility)

    return facilities


def extract_currency(sentence):
    currencies = {
        "تومن": "TOMAN",
        "تومان": "TOMAN",
        "ریال": "RIAL"
    }
    for word in sentence.split():
        if word in currencies:
            return currencies[word]
    return "TOMAN"


def extract_construction(message, comprehension_model, tokenizer):
    property_status = [
        "نوساز",
        "تازه ساخت",
        "نو ساز"
    ]  
    for word in message.split():
        if word in property_status:
            return word
    
    return extract_numbers_with_flags(comprehension_on_message(convert_persian_number(message), construction_question, comprehension_model, tokenizer)[0])
    


def information_retrieval_module(state, message, previous_questions_embeddings, model, comprehension_model, tokenizer):
    ## Greeting 
    if state == "GREETING": 
        previous_questions_embeddings, future_question = random_generator(greeting, previous_questions_embeddings, model)
        return future_question, "DISTRICT", ["سلام", "درود"], previous_questions_embeddings

    ## DISTRICT
    elif state == "DISTRICT": 
        previous_questions_embeddings, future_question = embedding_generator(model, previous_questions_embeddings, district)
        return future_question, "DIMENSIONS", ["سعادت آباد", "مجیدیه"], previous_questions_embeddings

    ## DIMENSIONS
    elif state == "DIMENSIONS": 
        print("NEIGHBORHOODS", find_neighbour(comprehension_on_message(message, district_question, comprehension_model, tokenizer)))
        previous_questions_embeddings, future_question = embedding_generator(model, previous_questions_embeddings, dimensions)
        return future_question, "COST", ["۱۲۰ متر", "۹۵"], previous_questions_embeddings

    # COST 
    elif state == "COST": 
        print("DIMENSIONS:", extract_numbers_with_flags(comprehension_on_message(convert_persian_number(message), dimensions_question, comprehension_model, tokenizer)[0]))
        previous_questions_embeddings, future_question = embedding_generator(model, previous_questions_embeddings, cost)
        return future_question, "ROOMS", ["حدود ۱ میلیارد", "بین ۲ تا ۳ میلیارد تومن"], previous_questions_embeddings


    # ROOMS
    elif state == "ROOMS":
        print("COST:", extract_numbers_with_flags(comprehension_on_message(convert_persian_number(message), cost_question, comprehension_model, tokenizer)[0]), "CURRENCY:", extract_currency(message))
        previous_questions_embeddings, future_question = embedding_generator(model, previous_questions_embeddings, rooms)
        return future_question, "FLOORS", ["یه خواب", "دو تا اتاق"], previous_questions_embeddings
    

    # FLOORS
    elif state == "FLOORS":
        print("ROOMS:", extract_numbers_with_flags(comprehension_on_message(convert_persian_number(message), rooms_question, comprehension_model, tokenizer)[0]))
        previous_questions_embeddings, future_question = embedding_generator(model, previous_questions_embeddings, floors)
        return future_question, "CONSTRUCTION", ["۴ طبقه", "۸ طبقه باشه"], previous_questions_embeddings    


    # CONSTRUCTION
    elif state == "CONSTRUCTION":
        print("FLOORS:", extract_numbers_with_flags(comprehension_on_message(convert_persian_number(message), floors_question, comprehension_model, tokenizer)[0]))
        previous_questions_embeddings, future_question = embedding_generator(model, previous_questions_embeddings, construction)
        return future_question, "FACILITIES", ["نوساز باشه", "۴ یا ۵ سال ساخت"], previous_questions_embeddings
    

    # FACILITIES
    elif state == "FACILITIES":
        print("CONSTRUCTION:", extract_construction(message, comprehension_model, tokenizer))
        previous_questions_embeddings, future_question = embedding_generator(model, previous_questions_embeddings, facilities)
        return future_question, "END", ["بالکن داشته باشه", "آسانسور داشته باشه"], previous_questions_embeddings
    

    # Thank user
    elif state == "END":
        print("FACILITIES:", extract_facilities(comprehension_on_message(message, facilities_question, comprehension_model, tokenizer)))
        return "امیدوارم تونسته باشم کمکت کنم.", "END2", ["خوب بود", "جالب نبود"], previous_questions_embeddings
    
    return
 
 
