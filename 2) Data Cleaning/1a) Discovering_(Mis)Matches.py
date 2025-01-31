import pandas as pd
import re
from difflib import SequenceMatcher
from nltk.stem import PorterStemmer

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Base list of unimportant words
UNIMPORTANT_WORDS_BASE = {
    "restaurant", "bar", "grill", "grille", "on", "the", "of", "&", "and", ",", "-",
    "sushi", "cafe", "café", "caffé", "caffe", "diner", "kitchen", "bistro", "pub", "house", "place",
    "shop", "thai", "japanese", "indian", "chinese", "italian", "mexican",
    "american", "french", "pizza", "burger", "steak", "seafood", "bbq", "wine",
    "beer", "ristorante", "barbeque", "market", "armenia", "lounge", "sport", "thru",
    "co.", "company", "place", "brewing", "store", "bakery", "buffet", "studio", "cafeteria",
    "mediterranean", "la", "las", "el", "suburban", "court", "latin", "garden", "famous", "old", "new", "chill", "ltd",
    "frozen", "yogurt", "express", "super", "coffee", "street", "style", "not", "your", "average", "foro", "romano",
    "hous", "coffe", "japanes", "steakhous", "ohca", "parlor", "india", "catrachita", "donut", "nashville",
    "mediteranean", "cupcake", "boutique", "Bar-B-Que", "tuscan", "plantation", "drive", "in", "vietnames", "food", "truck",
    "coffeehous", "romano", "china", "home", "station", "brewery", "ice", "cream", "caribbean", "taproom",
    "korean"
}

# Generate plural forms for words in the base list
def add_plural_forms(word_set):
    """Add plural forms to a set of words."""
    plural_set = set(word_set)
    for word in word_set:
        if not word.endswith("s"):
            plural_set.add(word + "s")
        if not word.endswith("es"):
            plural_set.add(word + "es")
    return plural_set

# Extend unimportant words with plural forms
UNIMPORTANT_WORDS = add_plural_forms(UNIMPORTANT_WORDS_BASE)

# List of countries and US states
COUNTRIES = {
    "united states", "canada", "mexico", "brazil", "united kingdom", "germany", "france", "italy", "spain",
    "australia", "india", "china", "japan", "south korea", "russia", "south africa", "egypt", "argentina",
    "saudi arabia", "turkey",
}

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
    "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}

WRITTEN_NUMBERS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion",
}

# Precompute stemmed versions of unimportant words
STEMMED_UNIMPORTANT_WORDS = {stemmer.stem(word) for word in UNIMPORTANT_WORDS}
STEMMED_COUNTRIES = {stemmer.stem(word) for word in COUNTRIES}
STEMMED_US_STATES = {stemmer.stem(word) for word in US_STATES}
STEMMED_WRITTEN_NUMBERS = {stemmer.stem(word) for word in WRITTEN_NUMBERS}

def clean_text(text, city, address):
    """Clean and normalize text by removing unimportant words and applying stemming."""
    if not isinstance(text, str):
        return ""

    if not isinstance(address, str):
        address = ""

    # Convert to lowercase
    text = text.lower()
    address = address.lower()

    # Tokenize text and address
    words = re.findall(r"\b[\w']+\b", text)
    address_words = set(re.findall(r"\b[\w']+\b", address))

    # Filter out unimportant words
    filtered_words = [
        word for word in words
        if stemmer.stem(word) not in STEMMED_UNIMPORTANT_WORDS
        and stemmer.stem(word) not in STEMMED_COUNTRIES
        and stemmer.stem(word) not in STEMMED_US_STATES
        and stemmer.stem(word) not in STEMMED_WRITTEN_NUMBERS
        and stemmer.stem(word) != stemmer.stem(city.lower())
        and stemmer.stem(word) not in {stemmer.stem(w) for w in address_words}
        and not word.isdigit()
    ]

    # Normalize words
    normalized_words = []
    for word in filtered_words:
        if word.endswith("'s") or word.endswith("s'"):
            word = word[:-2]
        elif word.endswith("s") and len(word) > 1:
            word = word[:-1]
        stemmed_word = stemmer.stem(word)
        normalized_words.append(stemmed_word)

    return " ".join(normalized_words)

def similarity_score(a, b):
    """Calculate similarity score between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def categorize_match(score):
    """Categorize similarity score into match levels."""
    if score >= 0.8:
        return "high match"
    elif score > 0.6:
        return "moderate match"
    elif score > 0.3:
        return "low match"
    else:
        return "no match"

# Load the dataset
data = pd.read_csv(
    '../1) Data Preperation/6) Fully_Merged_Database_Foursquare_Data.csv'
)

# Ensure columns exist
data["clean_name"] = data.apply(
    lambda row: clean_text(row.get("name", ""), row.get("city", ""), row.get("address", "")), axis=1
)
data["clean_foursquare_name"] = data.apply(
    lambda row: clean_text(row.get("foursquare_name", ""), row.get("city", ""), row.get("address", "")), axis=1
)

# Compute similarity and categorize
data["similarity_score"] = data.apply(
    lambda row: similarity_score(row["clean_name"], row["clean_foursquare_name"]), axis=1
)
data["match_category"] = data["similarity_score"].apply(categorize_match)

# Save results
data.to_csv("1b) Discovered_(Mis)Matches.csv", index=False)
print("Results saved")
