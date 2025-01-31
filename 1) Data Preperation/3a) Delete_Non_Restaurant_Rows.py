import pandas as pd

# Define the updated and cleaned list of cancellation keywords
cancellation_keywords = [
    "car repair", "auto services", "mechanic", "towing", "automotive", "vehicle maintenance", "roadside assistance",
    "spa", "salon", "waxing", "massage", "barber", "beauty", "skin care", "hair care", "facials", "nail",
    "hospital", "clinic", "medical center", "pharmacy", "wellness", "cleaning", "carpet cleaning", "maid services",
    "janitorial", "home services", "water cleaning", "sanitation", "pest control",
    "repair", "maintenance", "handyman", "hardware", "plumbing", "electrical", "appliance repair",
    "best buy", "home depot", "furniture", "home goods", "electronics", "office supplies",
    "festivals", "theatre", "entertainment", "events", "concerts", "park", "museum", "library",
    "photography", "artistry", "graphic design", "video production", "creative services",
    "fitness studio", "gym", "personal training", "yoga", "pilates",
    "rental", "leasing", "property management",
    "coaching", "tutoring", "mentoring", "workshops", "training",
    "post office", "mailing", "courier",
    "sport clips", "companies", "tour services",
    "fashion",
    "Department Stores", "Shopping", "Fashion", "Home & Garden", "Electronics", "Furniture Stores",
    "Doctors", "Traditional Chinese Medicine", "Naturopathic/Holistic", "Acupuncture", "Health & Medical", "Nutritionists",
    "Home Automation", "Local Services", "Home Theatre Installation", "TV Mounting",
    "Shipping Centers", "Notaries", "Mailbox Centers", "Printing Services",
    "Synagogues", "Religious Organizations",
    "Automotive", "Auto Parts & Supplies", "Auto Customization",
    "Vape Shops", "Tobacco Shops", "Personal Shopping", "Vitamins & Supplements",
    "Dance Wear", "Sports Wear", "Childrens Clothing", "Arts & Entertainment", "Social Clubs", "Performing Arts", "Sporting Goods", "Shoe Stores",
    "Mobile Phones", "Telecommunications", "Electronics", "Mobile Phone Accessories", "IT Services & Computer Repair",
    "Hair Salons", "Hair Extensions", "Beauty & Spas", "Wigs",
    "DUI Law", "Professional Services", "Lawyers", "Criminal Defense Law",
    "Laser Hair Removal", "Hair Removal", "Chiropractors", "Weight Loss Centers", "Sports Medicine", "Medical Spas",
    "Candle Stores", "Home Decor",
    "Event Planning & Services",
    "Uniforms",
    "Banks & Credit Unions", "Financial Services",
    "Tabletop Games", "Toy Stores", "Hobby Shops", "Comic Books",
    "Keys & Locksmiths",
    "Masonry/Concrete", "Gardeners", "Lawn Services", "Tree Services", "Landscape Architects", "Contractors", "Landscaping", "Irrigation", "Nurseries & Gardening",
    "Real Estate", "Mortgage Brokers", "Mortgage Lenders",
    "Fitness & Instruction", "Physical Therapy", "Active Life", "Trainers",
    "Women’s Clothing", "Men’s Clothing", "Accessories", "Jewelry", "Shoe Stores",
    "Printing Services", "Shipping Centers", "Couriers & Delivery Services",
    "Parenting Classes", "Maternity Wear", "Education", "Specialty Schools", "Laundry Services", "Child Care & Day Care", "Baby Gear & Furniture",
    "Hardware Stores", "Hot Tub & Pool", "Pool & Hot Tub Service",
    "Tires", "Auto Repair", "Oil Change Stations",
    "Medical Centers", "Diagnostic Services", "Orthopedists", "Spine Surgeons", "Diagnostic Imaging", "Pain Management", "Osteopathic Physicians",
    "Car Wash", "Auto Detailing",
    "Title Loans", "Installment Loans", "Check Cashing/Pay-day Loans",
    "Internal Medicine", "Oral Surgeons", "General Dentistry", "Dentists", "Cosmetic Dentists",
    "Walking Tours",
    "Performing Arts", "Cinema",
    "Art Classes", "Arts & Crafts", "Knitting Supplies", "Art Supplies",
    "Eyelash Service", "Beauty & Spas",
    "Apartments", "Roofing",
    "Plumbing",
    "Motorcycle Rental", "Tours", "Hiking", "Mountain Biking", "ATV Rentals/Tours", "RV Rental",
    "Machine Shops", "Auto Parts & Supplies",
    "Pet Services", "Pet Sitting", "Pets",
    "Local Flavor", "Bike Rentals", "Bikes", "Bike Repair/Maintenance",
    "Self Storage",
    "Art Galleries", "Piercing", "Tattoo", "Beauty & Spas",
    "Heating & Air Conditioning/HVAC",
    "Body Shops", "Wheel & Rim Repair",
    "Day Spas", "Cosmetics & Beauty Supply", "Nail Salons",
    "Pawn Shops", "Jewelry", "Gold Buyers", "Watches",
    "Golf",
    "Ranches", "Farms",
    "Movers", "Junk Removal & Hauling",
    "Skating Rinks",
    "Bridal", "Formal Wear", "Sewing & Alterations",
    "Musical Instruments & Teachers", "Scavenger Hunts", "Team Building Activities", "Employment Law", "Workers Compensation Law", "Barbers", "Cabinetry",
    "Zoos", "Botanical Gardens", "Dog Parks", "Taxis", "Boating", "Public Services & Government", "Police Departments",
    "Adult Entertainment", "Dance Clubs", "Car Rental", "Transportation", "Airlines", "Farmers Market",
    "Comedy Clubs", "Libraries", "Public Services & Government", "Hotels & Travel", "Hotel"
]

# Convert keywords to lowercase for case-insensitive matching
cancellation_keywords = [keyword.lower() for keyword in cancellation_keywords]

# Load the dataset
data = pd.read_csv("2b) Restructured_JSON_Database_Data.csv", low_memory=False)

# Strip leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# Function to check if any cancellation keyword is in the 'categories' column
def contains_cancellation_keyword(categories):
    if pd.isna(categories):
        return False
    # Split the categories into individual words and convert to lowercase
    category_list = [cat.strip().lower() for cat in categories.split(',')]
    # Check if any keyword is in the category list
    return any(keyword in category_list for keyword in cancellation_keywords)

# Filter out rows with matching keywords in the 'categories' column
filtered_data = data[~data['categories'].apply(contains_cancellation_keyword)]

# Save the filtered dataset to a new file
filtered_data.to_csv("3b) Deleted_Non_Restaurant_Rows.csv", index=False)
