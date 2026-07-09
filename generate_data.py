import pandas as pd
import numpy as np

# Maruwa (Electric Tricycle) Configuration
BATTERY_CAPACITY_KWH = 5.0 # 5 kWh battery
NOMINAL_VOLTAGE = 48.0 # 48V system
MAX_SPEED_KMH = 45.0
SAMPLING_RATE_S = 1 # 1 second intervals
TRIP_DURATION_S = 3600 * 2 # 2 hours of driving per trip
NUM_TRIPS = 10 # Generate 10 distinct trips to provide rich SoH and Temp variation

np.random.seed(42)

def generate_trip(trip_id, soh_start, ambient_temp):
    duration_seconds = TRIP_DURATION_S
    time_steps = np.arange(duration_seconds)
    
    # 1. Generate Speed Profile
    speed_kmh = np.zeros(duration_seconds)
    accel_ms2 = np.zeros(duration_seconds)
    target_speed = 0
    for i in range(1, duration_seconds):
        if i % 30 == 0:
            target_speed = np.clip(target_speed + np.random.normal(0, 10), 0, MAX_SPEED_KMH)
            if np.random.rand() < 0.2:
                target_speed = 0
        speed_diff = target_speed - speed_kmh[i-1]
        accel = np.clip(speed_diff * 0.1, -1.5, 1.0)
        speed_kmh[i] = np.clip(speed_kmh[i-1] + (accel * 3.6), 0, MAX_SPEED_KMH)
        accel_ms2[i] = accel

    # 2. Generate Road Slope
    slope_deg = np.zeros(duration_seconds)
    current_slope = 0
    for i in range(1, duration_seconds):
        if i % 60 == 0:
            current_slope = np.clip(current_slope + np.random.normal(0, 1.0), -5.0, 5.0)
        slope_deg[i] = current_slope
        
    # 3. Vehicle Load
    base_mass = 350
    passenger_load = np.zeros(duration_seconds)
    current_load = np.random.choice([0, 75, 150, 225])
    for i in range(duration_seconds):
        if i > 0 and i % 300 == 0 and speed_kmh[i] < 1.0:
            current_load = np.random.choice([0, 75, 150, 225])
        passenger_load[i] = current_load
        
    total_mass = base_mass + passenger_load

    # 4. Auxiliary Load
    aux_power_w = np.random.choice([50, 100, 150], size=duration_seconds)
    
    # 5. Physics and Electrical
    v_ms = speed_kmh / 3.6
    P_aero = 0.5 * 1.225 * 0.6 * 2.5 * (v_ms**3)
    P_roll = 0.015 * total_mass * 9.81 * v_ms
    P_accel = total_mass * accel_ms2 * v_ms
    P_slope = total_mass * 9.81 * np.sin(np.radians(slope_deg)) * v_ms
    
    efficiency = 0.85
    P_motor = (P_aero + P_roll + P_accel + P_slope) / efficiency
    P_motor[P_motor < 0] = P_motor[P_motor < 0] * 0.3 
    
    P_total = P_motor + aux_power_w
    
    # State tracking
    soc = np.zeros(duration_seconds)
    voltage = np.zeros(duration_seconds)
    current = np.zeros(duration_seconds)
    battery_temp = np.zeros(duration_seconds)
    soh = np.full(duration_seconds, soh_start) # SoH stays roughly constant over a 2 hour trip
    
    current_soc = 100.0
    # Battery capacity degrades with SoH
    actual_capacity_kwh = BATTERY_CAPACITY_KWH * (soh_start / 100.0)
    battery_capacity_ws = actual_capacity_kwh * 3600 * 1000
    
    # Internal resistance increases as SoH degrades and temp drops
    internal_resistance = 0.05 + ((100 - soh_start) * 0.001) + (max(0, 20 - ambient_temp) * 0.002)
    
    current_temp = ambient_temp
    
    for i in range(duration_seconds):
        v_base = 42.0 + (current_soc / 100.0) * (54.6 - 42.0)
        temp_current = P_total[i] / v_base
        v_actual = v_base - (temp_current * internal_resistance)
        v_actual = max(v_actual, 40.0)
        
        i_actual = P_total[i] / v_actual
        
        # Heat generation (I^2 * R) slowly raises battery temp
        heat_watts = (i_actual ** 2) * internal_resistance
        current_temp += (heat_watts * 0.0001) - ((current_temp - ambient_temp) * 0.001) # Heating - cooling
        
        voltage[i] = v_actual
        current[i] = i_actual
        soc[i] = current_soc
        battery_temp[i] = current_temp
        
        energy_used_ws = i_actual * v_actual * SAMPLING_RATE_S
        soc_drop = (energy_used_ws / battery_capacity_ws) * 100.0
        current_soc = max(0, current_soc - soc_drop)

    df = pd.DataFrame({
        'Trip_ID': trip_id,
        'Time_s': time_steps,
        'Speed_kmh': np.clip(speed_kmh, 0, None),
        'Acceleration_ms2': accel_ms2,
        'Load_kg': passenger_load,
        'Slope_deg': slope_deg,
        'Aux_Load_W': aux_power_w,
        'Battery_Temp_C': battery_temp,
        'SoH_Percent': soh,
        'Voltage_V': voltage,
        'Current_A': current,
        'SoC_Percent': np.clip(soc, 0, 100)
    })
    
    # Calculate target range: incorporates SoH and Temp penalties
    avg_speed_window = df['Speed_kmh'].rolling(300, min_periods=1).mean()
    base_consumption = np.where(avg_speed_window > 5, 80, 50)
    # Cold temperatures increase consumption penalty, SoH limits capacity
    temp_penalty = np.clip(1.0 + (20 - df['Battery_Temp_C']) * 0.01, 1.0, 1.3) 
    
    df['Target_Range_km'] = (df['SoC_Percent'] / 100.0) * actual_capacity_kwh * 1000 / (base_consumption * temp_penalty)

    # Add realistic sensor noise
    df['Voltage_V'] += np.random.normal(0, 0.1, duration_seconds)
    df['Current_A'] += np.random.normal(0, 0.5, duration_seconds)
    df['Speed_kmh'] += np.random.normal(0, 0.2, duration_seconds)
    df['SoC_Percent'] += np.random.normal(0, 0.05, duration_seconds)
    df['Battery_Temp_C'] += np.random.normal(0, 0.2, duration_seconds)

    return df

if __name__ == "__main__":
    print("Generating synthetic driving dataset with multiple trips for SoH/Temp variance...")
    
    all_trips = []
    # Simulate trips spanning different seasons and ages of the tricycle
    soh_values = [100.0, 98.5, 95.0, 92.0, 89.5, 85.0, 82.5, 80.0, 78.0, 75.0]
    temp_values = [25.0, 30.0, 15.0, 10.0, 5.0, 35.0, 20.0, -5.0, 12.0, 28.0]
    
    for i in range(NUM_TRIPS):
        print(f"Generating Trip {i+1}/{NUM_TRIPS} - SoH: {soh_values[i]}%, Temp: {temp_values[i]}C")
        trip_df = generate_trip(trip_id=i+1, soh_start=soh_values[i], ambient_temp=temp_values[i])
        all_trips.append(trip_df)
        
    final_df = pd.concat(all_trips, ignore_index=True)
    
    output_path = "maruwa_synthetic_data.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path} with {len(final_df)} total rows.")
