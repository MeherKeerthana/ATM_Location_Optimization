import numpy as np
import pandas as pd
import datetime
import random
import os
from scipy.spatial.distance import cdist

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# City Center Coordinate: Hyderabad (Ameerpet/Somajiguda area)
CITY_CENTER_LAT = 17.4129
CITY_CENTER_LNG = 78.4484

# Define 25 realistic Hyderabad zones
HYDERABAD_ZONES = [
    {"name": "HITEC City", "lat": 17.4483, "lng": 78.3741, "type": "commercial", "base_pop": 8000, "base_traffic": 25000, "income": 1800000, "rent": 95000, "metro": True, "mall_prob": 0.8},
    {"name": "Gachibowli", "lat": 17.4401, "lng": 78.3489, "type": "commercial", "base_pop": 7000, "base_traffic": 22000, "income": 1600000, "rent": 85000, "metro": True, "mall_prob": 0.6},
    {"name": "Charminar", "lat": 17.3616, "lng": 78.4747, "type": "dense_old_city", "base_pop": 25000, "base_traffic": 40000, "income": 450000, "rent": 40000, "metro": False, "mall_prob": 0.3},
    {"name": "Malakpet", "lat": 17.3756, "lng": 78.4998, "type": "dense_old_city", "base_pop": 18000, "base_traffic": 24000, "income": 500000, "rent": 30000, "metro": True, "mall_prob": 0.2},
    {"name": "Banjara Hills", "lat": 17.4156, "lng": 78.4347, "type": "commercial_premium", "base_pop": 6000, "base_traffic": 20000, "income": 2500000, "rent": 110000, "metro": False, "mall_prob": 0.9},
    {"name": "Jubilee Hills", "lat": 17.4300, "lng": 78.4080, "type": "commercial_premium", "base_pop": 5000, "base_traffic": 18000, "income": 2800000, "rent": 120000, "metro": True, "mall_prob": 0.8},
    {"name": "Secunderabad", "lat": 17.4399, "lng": 78.4983, "type": "commercial", "base_pop": 12000, "base_traffic": 28000, "income": 800000, "rent": 50000, "metro": True, "mall_prob": 0.5},
    {"name": "Kukatpally", "lat": 17.4834, "lng": 78.4084, "type": "commercial_dense", "base_pop": 22000, "base_traffic": 35000, "income": 900000, "rent": 65000, "metro": True, "mall_prob": 0.7},
    {"name": "Madhapur", "lat": 17.4485, "lng": 78.3908, "type": "commercial", "base_pop": 9000, "base_traffic": 26000, "income": 1400000, "rent": 80000, "metro": True, "mall_prob": 0.7},
    {"name": "Ameerpet", "lat": 17.4374, "lng": 78.4482, "type": "commercial_dense", "base_pop": 16000, "base_traffic": 38000, "income": 700000, "rent": 55000, "metro": True, "mall_prob": 0.4},
    {"name": "Kondapur", "lat": 17.4622, "lng": 78.3568, "type": "residential_high", "base_pop": 11000, "base_traffic": 18000, "income": 1200000, "rent": 45000, "metro": False, "mall_prob": 0.5},
    {"name": "Begumpet", "lat": 17.4375, "lng": 78.4618, "type": "commercial", "base_pop": 10000, "base_traffic": 22000, "income": 1100000, "rent": 60000, "metro": True, "mall_prob": 0.6},
    {"name": "Somajiguda", "lat": 17.4244, "lng": 78.4552, "type": "commercial", "base_pop": 8000, "base_traffic": 20000, "income": 1300000, "rent": 70000, "metro": False, "mall_prob": 0.6},
    {"name": "Koti", "lat": 17.3824, "lng": 78.4822, "type": "commercial_dense", "base_pop": 15000, "base_traffic": 30000, "income": 600000, "rent": 40000, "metro": False, "mall_prob": 0.3},
    {"name": "Nampally", "lat": 17.3892, "lng": 78.4682, "type": "transit_hub", "base_pop": 14000, "base_traffic": 32000, "income": 550000, "rent": 35000, "metro": True, "mall_prob": 0.2},
    {"name": "Dilsukhnagar", "lat": 17.3688, "lng": 78.5247, "type": "commercial_dense", "base_pop": 20000, "base_traffic": 34000, "income": 750000, "rent": 50000, "metro": True, "mall_prob": 0.5},
    {"name": "Tarnaka", "lat": 17.4267, "lng": 78.5375, "type": "residential", "base_pop": 10000, "base_traffic": 14000, "income": 700000, "rent": 30000, "metro": True, "mall_prob": 0.3},
    {"name": "Miyapur", "lat": 17.4966, "lng": 78.3598, "type": "residential", "base_pop": 13000, "base_traffic": 16000, "income": 750000, "rent": 28000, "metro": True, "mall_prob": 0.4},
    {"name": "LB Nagar", "lat": 17.3578, "lng": 78.5489, "type": "residential", "base_pop": 15000, "base_traffic": 20000, "income": 800000, "rent": 35000, "metro": True, "mall_prob": 0.4},
    {"name": "Uppal", "lat": 17.3985, "lng": 78.5562, "type": "residential", "base_pop": 14000, "base_traffic": 18000, "income": 700000, "rent": 30000, "metro": True, "mall_prob": 0.3},
    {"name": "Mehdipatnam", "lat": 17.3958, "lng": 78.4312, "type": "transit_hub", "base_pop": 17000, "base_traffic": 30000, "income": 650000, "rent": 40000, "metro": False, "mall_prob": 0.4},
    {"name": "Tolichowki", "lat": 17.3992, "lng": 78.4087, "type": "residential", "base_pop": 12000, "base_traffic": 16000, "income": 700000, "rent": 30000, "metro": False, "mall_prob": 0.3},
    {"name": "Manikonda", "lat": 17.3986, "lng": 78.3789, "type": "residential_high", "base_pop": 10000, "base_traffic": 15000, "income": 1100000, "rent": 35000, "metro": False, "mall_prob": 0.5},
    {"name": "Nizampet", "lat": 17.5144, "lng": 78.3842, "type": "residential", "base_pop": 11000, "base_traffic": 12000, "income": 650000, "rent": 25000, "metro": False, "mall_prob": 0.3},
    {"name": "Pragathi Nagar", "lat": 17.5025, "lng": 78.3950, "type": "residential", "base_pop": 12000, "base_traffic": 13000, "income": 700000, "rent": 25000, "metro": False, "mall_prob": 0.3}
]

def lat_lng_distance(lat1, lng1, lat2, lng2):
    """Calculate approximate distance in kilometers between two lat/lng coordinates"""
    lat_dist = (lat1 - lat2) * 111.0
    lng_dist = (lng1 - lng2) * 111.0 * np.cos(np.radians(lat1))
    return np.sqrt(lat_dist**2 + lng_dist**2)

def is_in_water(lat, lng):
    """Check if lat/lng falls inside known Hyderabad lakes or water bodies"""
    # Hussain Sagar Lake
    if 17.4120 <= lat <= 17.4380 and 78.4580 <= lng <= 78.4800:
        return True
    # Durgam Cheruvu (Secret Lake)
    if 17.4280 <= lat <= 17.4450 and 78.3780 <= lng <= 78.3950:
        return True
    # Gachibowli Bio-Diversity / Khajaguda / Malkam Cheruvu Lake
    if 17.4200 <= lat <= 17.4360 and 78.3450 <= lng <= 78.3600:
        return True
    # Kukatpally / Moosapet IDL / Pragathi Nagar Cheruvu
    if 17.4720 <= lat <= 17.5100 and 78.3950 <= lng <= 78.4220:
        return True
    # Saroornagar Lake
    if 17.3480 <= lat <= 17.3680 and 78.5220 <= lng <= 78.5420:
        return True
    # Mir Alam Tank
    if 17.3450 <= lat <= 17.3650 and 78.4400 <= lng <= 78.4600:
        return True
    # Osman Sagar & Himayat Sagar
    if 17.3650 <= lat <= 17.4150 and 78.2650 <= lng <= 78.3250:
        return True
    if 17.2950 <= lat <= 17.3480 and 78.3350 <= lng <= 78.3900:
        return True
    return False

sanitize_rng = np.random.RandomState(999)

def sanitize_coordinate(lat, lng, zone_name):
    """Ensure coordinates do not land in water bodies by shifting them onto dry land in the zone using dedicated RNG"""
    attempts = 0
    curr_lat, curr_lng = lat, lng
    while is_in_water(curr_lat, curr_lng) and attempts < 50:
        attempts += 1
        if zone_name == "Somajiguda":
            curr_lat = 17.4244 + sanitize_rng.uniform(-0.003, 0.003)
            curr_lng = 78.4530 + sanitize_rng.uniform(-0.003, 0.003)
        elif zone_name == "Begumpet":
            curr_lat = 17.4410 + sanitize_rng.uniform(-0.002, 0.002)
            curr_lng = 78.4550 + sanitize_rng.uniform(-0.003, 0.003)
        elif zone_name == "Gachibowli":
            curr_lat = 17.4410 + sanitize_rng.uniform(-0.002, 0.002)
            curr_lng = 78.3470 + sanitize_rng.uniform(-0.003, 0.003)
        elif zone_name in ["HITEC City", "Madhapur"]:
            curr_lat = 17.4480 + sanitize_rng.uniform(-0.002, 0.002)
            curr_lng = 78.3750 + sanitize_rng.uniform(-0.003, 0.003)
        elif zone_name in ["Kukatpally", "Pragathi Nagar", "Nizampet"]:
            curr_lat = 17.4950 + sanitize_rng.uniform(-0.003, 0.003)
            curr_lng = 78.3960 + sanitize_rng.uniform(-0.003, 0.003)
        elif zone_name in ["Dilsukhnagar", "LB Nagar"]:
            curr_lat = 17.3688 + sanitize_rng.uniform(-0.003, 0.003)
            curr_lng = 78.5200 + sanitize_rng.uniform(-0.003, 0.003)
        else:
            curr_lat = curr_lat + sanitize_rng.uniform(-0.008, 0.008)
            curr_lng = curr_lng + sanitize_rng.uniform(-0.008, 0.008)
    return curr_lat, curr_lng

def generate_city_grid(size=1500):
    """
    Generate demographic grid cells representing neighborhoods across the 25 Hyderabad zones.
    """
    data = []
    cells_per_zone = size // len(HYDERABAD_ZONES)
    
    cell_idx = 0
    for zone in HYDERABAD_ZONES:
        for _ in range(cells_per_zone):
            # Sample coordinates around the zone center
            lat = zone['lat'] + np.random.normal(0, 0.006)
            lng = zone['lng'] + np.random.normal(0, 0.006)
            lat, lng = sanitize_coordinate(lat, lng, zone['name'])
            
            # Demographic attributes based on zone features + local jitter (with independent income variation)
            pop_density = max(500, int(zone['base_pop'] + np.random.normal(0, zone['base_pop'] * 0.15)))
            foot_traffic = max(200, int(zone['base_traffic'] + np.random.normal(0, zone['base_traffic'] * 0.2)))
            avg_income = max(100000, int(zone['income'] * np.random.uniform(0.75, 1.25) + np.random.normal(0, 50000)))
            
            # Commercial activity index (0 to 100)
            comm_val = 80.0 if zone['type'] in ['commercial', 'commercial_premium', 'commercial_dense'] else 30.0
            commercial_activity = max(5.0, min(100.0, comm_val + np.random.normal(0, 10.0)))
            
            # Metro station footfall (continuous, graduated range across cells)
            if zone['metro']:
                nearby_metro_footfall = max(500, int(np.random.normal(12000, 3500)))
            else:
                nearby_metro_footfall = max(200, int(np.random.normal(2500, 1000)))
                
            # Market / mall proximity score (0 to 100)
            mall_score = max(0, min(100, int(zone['mall_prob'] * 100 + np.random.normal(0, 15))))
            
            data.append({
                'cell_id': cell_idx,
                'zone_name': zone['name'],
                'latitude': lat,
                'longitude': lng,
                'pop_density': pop_density,
                'foot_traffic': foot_traffic,
                'avg_income': avg_income,
                'commercial_activity': round(commercial_activity, 2),
                'nearby_metro_footfall': nearby_metro_footfall,
                'market_mall_proximity': mall_score
            })
            cell_idx += 1
            
    return pd.DataFrame(data)

def generate_atms(grid_df, count_own=400, count_competitor=150):
    """
    Generate existing own ATMs and competitor ATMs, placed logically in higher traffic cells.
    """
    # Sort grid by foot traffic
    sorted_grid = grid_df.sort_values(by='foot_traffic', ascending=False).reset_index(drop=True)

    # Pool sizes scale with the requested counts so larger datasets still get a
    # sensible "biased toward high foot-traffic cells" sampling pool instead of
    # hard-coded pools sized for the old 400/150 defaults.
    own_pool_size = min(len(sorted_grid), int(count_own * 1.2) + 80)
    own_indices = random.sample(range(own_pool_size), count_own)

    # Competitor sites are chosen BEFORE the own-ATM loop so that competitor
    # proximity can be used as a real, causal, non-linear input to the
    # transaction-generation formula below (previously this feature was fed
    # to the ML model but had no actual effect on the generated data).
    remaining_pool = [x for x in range(min(len(sorted_grid), own_pool_size + count_competitor + 200)) if x not in own_indices]
    comp_indices = random.sample(remaining_pool, count_competitor)
    comp_cells = [sorted_grid.iloc[idx] for idx in comp_indices]
    comp_cell_coords = np.array([[c['latitude'], c['longitude']] for c in comp_cells])

    own_atms = []
    for i, idx in enumerate(own_indices):
        cell = sorted_grid.iloc[idx]
        atm_id = f"ATM_{i+1:03d}"
        
        uptime_pct = round(np.random.uniform(97.5, 99.9), 2)
        cashout_rate_pct = round(np.random.uniform(0.5, 3.0), 2)
        
        # Get matching zone details to set rent
        matching_zone = [z for z in HYDERABAD_ZONES if z['name'] == cell['zone_name']][0]
        rent_cost = int(matching_zone['rent'] + np.random.normal(0, matching_zone['rent'] * 0.1))
        rent_cost = max(10000, min(150000, rent_cost))
        
        # Transactions based on realistic non-linear factors
        ft = cell['foot_traffic']
        pop = cell['pop_density']
        comm = cell['commercial_activity']
        metro = cell['nearby_metro_footfall']
        mall = cell['market_mall_proximity']
        inc = cell['avg_income']
        
        # 1. Hard saturation curve (sharp logistic cap past 15k foot traffic)
        sat_ft = 50.0 / (1.0 + np.exp(-(ft - 15000.0) / 1500.0))
        
        # 2. Multiplicative conditional interactions (tree models capture naturally, LinearRegression cannot)
        inter_cond1 = 45.0 if (mall > 12.0 and comm > 12.0) else (20.0 if (mall > 6.0 and comm > 6.0) else 0.0)
        inter_cond2 = 14.0 if (metro > 10000.0 and ft > 15000.0) else 0.0
        
        # 3. Discontinuous regime-switch / threshold jumps
        regime_comm = 35.0 if comm > 14.0 else (15.0 if comm > 7.0 else 0.0)
        regime_inc = 12.0 if inc > 750000 else 0.0
        
        # 4. Spatial hub decay effect (exponential decay from HITECH City hub)
        hub_dist = lat_lng_distance(cell['latitude'], cell['longitude'], 17.4435, 78.3772)
        hub_decay = 40.0 * np.exp(-hub_dist / 4.0)
        
        # Base logarithmic terms for population density and income
        term_pop = np.log1p(pop / 1000.0) * 14.0
        term_inc = np.log1p(inc / 10000.0) * 4.5
        
        # 5. Competitor cannibalization (non-linear, distance-based decay).
        # A rival ATM sitting very close by (a few hundred metres) siphons off
        # a meaningful share of the same cash-withdrawal demand; the effect
        # fades out fast as distance grows, so this is a steep exponential
        # decay rather than a linear penalty. This is what makes
        # `dist_to_nearest_competitor` an actually-causal feature for the ML
        # model to learn, instead of pure noise.
        dist_to_comp = cdist(
            np.array([[cell['latitude'], cell['longitude']]]), comp_cell_coords,
            lambda u, v: lat_lng_distance(u[0], u[1], v[0], v[1])
        ).min()
        cannibal_penalty = 22.0 * np.exp(-dist_to_comp / 0.35)
        
        composite = sat_ft + inter_cond1 + inter_cond2 + regime_comm + regime_inc + hub_decay + term_pop + term_inc - cannibal_penalty
        
        avg_daily_tx = max(15, int(composite * (uptime_pct / 100.0) * (1.0 - cashout_rate_pct / 200.0) + np.random.normal(0, 1.5)))
        
        own_lat, own_lng = sanitize_coordinate(cell['latitude'] + np.random.normal(0, 0.0001), cell['longitude'] + np.random.normal(0, 0.0001), cell['zone_name'])
        own_atms.append({
            'atm_id': atm_id,
            'zone_name': cell['zone_name'],
            'area_type': matching_zone['type'],
            'site_type': random.choice(["Metro Station Exit", "Shopping Mall Lobby", "Supermarket Entrance", "IT Park Food Court", "Fuel Station Complex", "Hospital Plaza"]),
            'months_in_service': random.randint(1, 120),
            'latitude': own_lat,
            'longitude': own_lng,
            'avg_daily_transactions': avg_daily_tx,
            'uptime_pct': uptime_pct,
            'cashout_rate_pct': cashout_rate_pct,
            'rent_cost': rent_cost,
            'foot_traffic': cell['foot_traffic'],
            'pop_density': cell['pop_density'],
            'avg_income': cell['avg_income'],
            'commercial_activity': cell['commercial_activity'],
            'nearby_metro_footfall': cell['nearby_metro_footfall'],
            'market_mall_proximity': cell['market_mall_proximity']
        })
        
    # Build competitor ATM records (sites were already picked above, before the
    # own-ATM loop, so the cannibalization term could use them)
    banks = ["SBI ATM", "HDFC Bank ATM", "ICICI Bank ATM", "Axis Bank ATM", "Kotak ATM"]
    comp_atms = []
    for i, idx in enumerate(comp_indices):
        cell = sorted_grid.iloc[idx]
        comp_id = f"COMP_{i+1:03d}"
        comp_lat, comp_lng = sanitize_coordinate(cell['latitude'] + np.random.normal(0, 0.0001), cell['longitude'] + np.random.normal(0, 0.0001), cell['zone_name'])
        comp_atms.append({
            'competitor_id': comp_id,
            'zone_name': cell['zone_name'],
            'bank_name': random.choice(banks),
            'latitude': comp_lat,
            'longitude': comp_lng,
            'foot_traffic': cell['foot_traffic'],
            'pop_density': cell['pop_density'],
            'commercial_activity': cell['commercial_activity'],
            'nearby_metro_footfall': cell['nearby_metro_footfall'],
            'market_mall_proximity': cell['market_mall_proximity']
        })
        
    return pd.DataFrame(own_atms), pd.DataFrame(comp_atms)

def generate_candidates(grid_df, own_atms_df, comp_atms_df, count=150):
    """
    Generate candidate locations where new ATMs could be leased in Hyderabad.
    """
    candidates = []
    site_types = ["Metro Station Exit", "Shopping Mall Lobby", "Supermarket Entrance", "IT Park Food Court", "Fuel Station Complex", "Hospital Plaza"]
    
    # Sort grid by foot traffic
    sorted_grid = grid_df.sort_values(by='foot_traffic', ascending=False).reset_index(drop=True)
    
    candidate_added = 0
    grid_idx = 80 # skip absolute highest to keep it realistic
    
    while candidate_added < count and grid_idx < len(sorted_grid):
        cell = sorted_grid.iloc[grid_idx]
        grid_idx += 5 # spread them out spatially
        
        # Check distance to closest own ATM
        min_own_dist = min([lat_lng_distance(cell['latitude'], cell['longitude'], row['latitude'], row['longitude']) 
                             for idx, row in own_atms_df.iterrows()])
        min_comp_dist = min([lat_lng_distance(cell['latitude'], cell['longitude'], row['latitude'], row['longitude']) 
                              for idx, row in comp_atms_df.iterrows()])
        
        # We want candidates that aren't right next to another own ATM (at least 200m away)
        if min_own_dist > 0.2:
            cand_id = f"CAND_{candidate_added+1:03d}"
            site_type = random.choice(site_types)
            
            matching_zone = [z for z in HYDERABAD_ZONES if z['name'] == cell['zone_name']][0]
            rent_cost = int(matching_zone['rent'] * 0.95 + np.random.normal(0, matching_zone['rent'] * 0.05))
            rent_cost = max(10000, min(140000, rent_cost))
            
            cand_lat, cand_lng = sanitize_coordinate(cell['latitude'], cell['longitude'], cell['zone_name'])
            candidates.append({
                'candidate_id': cand_id,
                'zone_name': cell['zone_name'],
                'area_type': matching_zone['type'],
                'months_in_service': 0,
                'site_type': site_type,
                'name': f"{matching_zone['name']} {site_type} {cand_id.split('_')[1]}",
                'latitude': cand_lat,
                'longitude': cand_lng,
                'foot_traffic': cell['foot_traffic'],
                'pop_density': cell['pop_density'],
                'avg_income': cell['avg_income'],
                'commercial_activity': cell['commercial_activity'],
                'nearby_metro_footfall': cell['nearby_metro_footfall'],
                'market_mall_proximity': cell['market_mall_proximity'],
                'rent_cost': rent_cost,
                'dist_to_nearest_competitor': min_comp_dist,
                'dist_to_nearest_own_atm': min_own_dist
            })
            candidate_added += 1
            
    return pd.DataFrame(candidates)

def generate_transaction_logs(own_atms_df, days=30):
    """
    Generate detailed synthetic transaction logs with Indian withdrawal denominations,
    holiday effects, and festive spikes.
    To avoid huge load times, we generate logs for a random sample of 30 own ATMs.
    """
    logs = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    tx_types = ["Withdrawal", "Deposit", "Balance Inquiry", "Transfer"]
    tx_weights = [0.65, 0.20, 0.10, 0.05]
    card_types = ["Debit - RuPay", "Debit - Visa", "Debit - Mastercard", "Credit - RuPay", "Credit - Visa"]
    card_weights = [0.40, 0.30, 0.20, 0.05, 0.05]
    
    # Hourly weights (Simulates peaks at 12-14 lunch and 17-21 evening)
    hourly_weights = [
        0.01, 0.005, 0.002, 0.002, 0.005, 0.015,  # 00:00 - 05:00
        0.03, 0.05, 0.07, 0.06, 0.06, 0.07,       # 06:00 - 11:00
        0.09, 0.09, 0.07, 0.06, 0.07, 0.09,       # 12:00 - 17:00
        0.09, 0.08, 0.05, 0.03, 0.02, 0.015       # 18:00 - 23:00
    ]
    hourly_weights = np.array(hourly_weights)
    hourly_weights = hourly_weights / hourly_weights.sum()
    
    # Day-of-week weights (higher on Fri/Sat/Sun)
    dow_weights = [0.9, 0.9, 0.95, 1.0, 1.25, 1.3, 1.1]
    
    # Sub-sample 30 ATMs to generate logs for, keeping it fast and lightweight
    sampled_atms = own_atms_df.sample(n=30, random_state=42)
    
    # Pre-select some holiday indices (e.g., day 10 is Independence Day, day 25 is Diwali)
    holidays = [10, 25]
    festival_days = list(range(20, 24)) # Festive week
    
    for _, atm in sampled_atms.iterrows():
        atm_id = atm['atm_id']
        daily_avg = atm['avg_daily_transactions']
        
        for day in range(days):
            current_date = start_date + datetime.timedelta(days=day)
            day_of_week = current_date.weekday()
            
            # Holiday and festive factors
            multiplier = 1.0
            is_holiday_flag = False
            is_festival_flag = False
            
            if day in holidays:
                multiplier *= 1.4
                is_holiday_flag = True
            if day in festival_days:
                multiplier *= 1.3
                is_festival_flag = True
                
            # Adjust daily volume based on factors & random shock
            target_tx_count = int(daily_avg * dow_weights[day_of_week] * multiplier + np.random.normal(0, daily_avg * 0.1))
            target_tx_count = max(3, target_tx_count)
            
            # Sample hour for each transaction
            hours = np.random.choice(24, size=target_tx_count, p=hourly_weights)
            
            for hr in hours:
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                tx_time = current_date.replace(hour=hr, minute=minute, second=second)
                
                tx_type = np.random.choice(tx_types, p=tx_weights)
                card = np.random.choice(card_types, p=card_weights)
                
                # Transaction Amount in Indian withdrawal denominations (₹100/200/500/2000 multiples)
                if tx_type == "Withdrawal":
                    amount = random.choice([100, 200, 500, 1000, 1500, 2000, 3000, 5000, 10000])
                elif tx_type == "Deposit":
                    amount = random.choice([500, 1000, 2000, 5000, 10000, 20000, 49000]) # under 50k cash deposit limit
                else:
                    amount = 0
                    
                is_failed = random.random() > (atm['uptime_pct'] / 100.0)
                status = "Failed" if is_failed else "Success"
                error_msg = random.choice(["Network Timeout", "Hardware Error", "Insufficient Cash"]) if is_failed else None
                
                logs.append({
                    'timestamp': tx_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'atm_id': atm_id,
                    'transaction_type': tx_type,
                    'amount': amount,
                    'card_type': card,
                    'status': status,
                    'error_message': error_msg,
                    'is_holiday': 1 if is_holiday_flag else 0,
                    'is_festival': 1 if is_festival_flag else 0
                })
                
    logs_df = pd.DataFrame(logs)
    logs_df = logs_df.sort_values(by='timestamp').reset_index(drop=True)
    return logs_df

def run_data_generation():
    """Generate all data files and save them as CSV inside a data/ directory"""
    np.random.seed(42)
    random.seed(42)
    print("Generating demographic grid for Hyderabad...")
    grid = generate_city_grid(4200)
    
    print("Generating Hyderabad ATM locations (scaled)...")
    own_atms, comp_atms = generate_atms(grid, 1000, 300)
    
    print("Generating candidate locations (scaled)...")
    candidates = generate_candidates(grid, own_atms, comp_atms, 300)
    
    print("Generating transaction logs...")
    tx_logs = generate_transaction_logs(own_atms, 30)
    
    # Save files to workspace directory
    os.makedirs('data', exist_ok=True)
    grid.to_csv('data/demographics.csv', index=False)
    own_atms.to_csv('data/atms_own.csv', index=False)
    comp_atms.to_csv('data/atms_competitor.csv', index=False)
    candidates.to_csv('data/candidates.csv', index=False)
    tx_logs.to_csv('data/transaction_logs.csv', index=False)
    print("Data generation complete! Saved in 'data/' directory.")

if __name__ == "__main__":
    run_data_generation()