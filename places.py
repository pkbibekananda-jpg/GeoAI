"""Places used by GeoAI: Hungarian cities and major world cities (approximate city-centre coordinates)."""

HUNGARY = {
    "Budapest": (47.4979, 19.0402), "Debrecen": (47.5316, 21.6273), "Szeged": (46.2530, 20.1414),
    "Miskolc": (48.1035, 20.7784), "Pécs": (46.0727, 18.2323), "Győr": (47.6875, 17.6504),
    "Nyíregyháza": (47.9554, 21.7167), "Kecskemét": (46.8964, 19.6897),
    "Székesfehérvár": (47.1860, 18.4221), "Szombathely": (47.2307, 16.6218),
    "Szolnok": (47.1621, 20.1825), "Tatabánya": (47.5692, 18.4048), "Kaposvár": (46.3594, 17.7968),
    "Veszprém": (47.0933, 17.9115), "Békéscsaba": (46.6736, 21.0877), "Zalaegerszeg": (46.8417, 16.8416),
    "Sopron": (47.6817, 16.5845), "Eger": (47.9025, 20.3772), "Nagykanizsa": (46.4590, 16.9897),
    "Dunaújváros": (46.9619, 18.9355),
}

# name: (lat, lon, country)
WORLD = {
    # Europe
    "London": (51.5074, -0.1278, "United Kingdom"), "Paris": (48.8566, 2.3522, "France"),
    "Berlin": (52.5200, 13.4050, "Germany"), "Madrid": (40.4168, -3.7038, "Spain"),
    "Rome": (41.9028, 12.4964, "Italy"), "Milan": (45.4642, 9.1900, "Italy"),
    "Vienna": (48.2082, 16.3738, "Austria"), "Prague": (50.0755, 14.4378, "Czechia"),
    "Warsaw": (52.2297, 21.0122, "Poland"), "Krakow": (50.0647, 19.9450, "Poland"),
    "Bucharest": (44.4268, 26.1025, "Romania"), "Belgrade": (44.7866, 20.4489, "Serbia"),
    "Sofia": (42.6977, 23.3219, "Bulgaria"), "Athens": (37.9838, 23.7275, "Greece"),
    "Istanbul": (41.0082, 28.9784, "Türkiye"), "Kyiv": (50.4501, 30.5234, "Ukraine"),
    "Moscow": (55.7558, 37.6173, "Russia"), "Stockholm": (59.3293, 18.0686, "Sweden"),
    "Oslo": (59.9139, 10.7522, "Norway"), "Copenhagen": (55.6761, 12.5683, "Denmark"),
    "Helsinki": (60.1699, 24.9384, "Finland"), "Amsterdam": (52.3676, 4.9041, "Netherlands"),
    "Brussels": (50.8503, 4.3517, "Belgium"), "Zurich": (47.3769, 8.5417, "Switzerland"),
    "Lisbon": (38.7223, -9.1393, "Portugal"), "Dublin": (53.3498, -6.2603, "Ireland"),
    "Bratislava": (48.1486, 17.1077, "Slovakia"), "Zagreb": (45.8150, 15.9819, "Croatia"),
    "Ljubljana": (46.0569, 14.5058, "Slovenia"), "Sarajevo": (43.8563, 18.4131, "Bosnia and Herzegovina"),
    "Skopje": (41.9981, 21.4254, "North Macedonia"), "Cluj-Napoca": (46.7712, 23.6236, "Romania"),
    # Asia
    "Delhi": (28.6139, 77.2090, "India"), "Mumbai": (19.0760, 72.8777, "India"),
    "Kolkata": (22.5726, 88.3639, "India"), "Chennai": (13.0827, 80.2707, "India"),
    "Bengaluru": (12.9716, 77.5946, "India"), "Lucknow": (26.8467, 80.9462, "India"),
    "Dhaka": (23.8103, 90.4125, "Bangladesh"), "Chittagong": (22.3569, 91.7832, "Bangladesh"),
    "Lahore": (31.5204, 74.3587, "Pakistan"), "Karachi": (24.8607, 67.0011, "Pakistan"),
    "Kathmandu": (27.7172, 85.3240, "Nepal"), "Colombo": (6.9271, 79.8612, "Sri Lanka"),
    "Beijing": (39.9042, 116.4074, "China"), "Shanghai": (31.2304, 121.4737, "China"),
    "Guangzhou": (23.1291, 113.2644, "China"), "Chengdu": (30.5728, 104.0668, "China"),
    "Wuhan": (30.5928, 114.3055, "China"), "Xi'an": (34.3416, 108.9398, "China"),
    "Hong Kong": (22.3193, 114.1694, "China"), "Taipei": (25.0330, 121.5654, "Taiwan"),
    "Seoul": (37.5665, 126.9780, "South Korea"), "Tokyo": (35.6762, 139.6503, "Japan"),
    "Osaka": (34.6937, 135.5023, "Japan"), "Ulaanbaatar": (47.8864, 106.9057, "Mongolia"),
    "Bangkok": (13.7563, 100.5018, "Thailand"), "Chiang Mai": (18.7883, 98.9853, "Thailand"),
    "Hanoi": (21.0278, 105.8342, "Vietnam"), "Ho Chi Minh City": (10.8231, 106.6297, "Vietnam"),
    "Jakarta": (-6.2088, 106.8456, "Indonesia"), "Pekanbaru": (0.5071, 101.4478, "Indonesia"),
    "Palembang": (-2.9761, 104.7754, "Indonesia"), "Kuala Lumpur": (3.1390, 101.6869, "Malaysia"),
    "Singapore": (1.3521, 103.8198, "Singapore"), "Manila": (14.5995, 120.9842, "Philippines"),
    "Yangon": (16.8409, 96.1735, "Myanmar"), "Tehran": (35.6892, 51.3890, "Iran"),
    "Baghdad": (33.3152, 44.3661, "Iraq"), "Riyadh": (24.7136, 46.6753, "Saudi Arabia"),
    "Dubai": (25.2048, 55.2708, "United Arab Emirates"), "Doha": (25.2854, 51.5310, "Qatar"),
    "Kuwait City": (29.3759, 47.9774, "Kuwait"), "Tashkent": (41.2995, 69.2401, "Uzbekistan"),
    "Almaty": (43.2220, 76.8512, "Kazakhstan"), "Kabul": (34.5553, 69.2075, "Afghanistan"),
    # Africa
    "Cairo": (30.0444, 31.2357, "Egypt"), "Lagos": (6.5244, 3.3792, "Nigeria"),
    "Abuja": (9.0765, 7.3986, "Nigeria"), "Kano": (12.0022, 8.5920, "Nigeria"),
    "Accra": (5.6037, -0.1870, "Ghana"), "Dakar": (14.7167, -17.4677, "Senegal"),
    "N'Djamena": (12.1348, 15.0557, "Chad"), "Niamey": (13.5116, 2.1254, "Niger"),
    "Bamako": (12.6392, -8.0029, "Mali"), "Khartoum": (15.5007, 32.5599, "Sudan"),
    "Addis Ababa": (8.9806, 38.7578, "Ethiopia"), "Nairobi": (-1.2921, 36.8219, "Kenya"),
    "Kampala": (0.3476, 32.5825, "Uganda"), "Kinshasa": (-4.4419, 15.2663, "DR Congo"),
    "Luanda": (-8.8390, 13.2894, "Angola"), "Johannesburg": (-26.2041, 28.0473, "South Africa"),
    "Cape Town": (-33.9249, 18.4241, "South Africa"), "Casablanca": (33.5731, -7.5898, "Morocco"),
    "Algiers": (36.7538, 3.0588, "Algeria"), "Tunis": (36.8065, 10.1815, "Tunisia"),
    # Americas
    "New York": (40.7128, -74.0060, "United States"), "Los Angeles": (34.0522, -118.2437, "United States"),
    "Chicago": (41.8781, -87.6298, "United States"), "Houston": (29.7604, -95.3698, "United States"),
    "Denver": (39.7392, -104.9903, "United States"), "Seattle": (47.6062, -122.3321, "United States"),
    "Phoenix": (33.4484, -112.0740, "United States"), "Toronto": (43.6532, -79.3832, "Canada"),
    "Vancouver": (49.2827, -123.1207, "Canada"), "Mexico City": (19.4326, -99.1332, "Mexico"),
    "Guatemala City": (14.6349, -90.5069, "Guatemala"), "Havana": (23.1136, -82.3666, "Cuba"),
    "Bogotá": (4.7110, -74.0721, "Colombia"), "Lima": (-12.0464, -77.0428, "Peru"),
    "Quito": (-0.1807, -78.4678, "Ecuador"), "Santiago": (-33.4489, -70.6693, "Chile"),
    "Buenos Aires": (-34.6037, -58.3816, "Argentina"), "São Paulo": (-23.5505, -46.6333, "Brazil"),
    "Rio de Janeiro": (-22.9068, -43.1729, "Brazil"), "Manaus": (-3.1190, -60.0217, "Brazil"),
    "Porto Velho": (-8.7612, -63.9004, "Brazil"), "La Paz": (-16.4897, -68.1193, "Bolivia"),
    # Oceania
    "Sydney": (-33.8688, 151.2093, "Australia"), "Melbourne": (-37.8136, 144.9631, "Australia"),
    "Perth": (-31.9505, 115.8605, "Australia"), "Auckland": (-36.8485, 174.7633, "New Zealand"),
}


def all_places():
    """name -> (lat, lon, country) for every built-in place."""
    out = {n: (la, lo, "Hungary") for n, (la, lo) in HUNGARY.items()}
    out.update(WORLD)
    return out
