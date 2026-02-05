"""Ground truth answers for ATNF-Chat test questions.

Uses psrqpy to query the ATNF catalogue directly and compute
verifiable answers for a set of benchmark questions.
"""

import numpy as np
import psrqpy

# Load the full catalogue
print("Loading ATNF catalogue...")
query = psrqpy.QueryATNF()
df = query.pandas
print(f"Catalogue version: {query.get_version}")
print(f"Total pulsars: {len(df)}")
print("=" * 70)


# --- Q1: North star metric ---
print("\nQ1: How many millisecond pulsars have orbital periods less than 1 day?")
msp_short_orbit = df[(df["P0"] < 0.03) & (df["PB"].notna()) & (df["PB"] < 1.0)]
print(f"  Answer: {len(msp_short_orbit)}")


# --- Q2: Simple count - fast pulsars ---
print("\nQ2: How many pulsars have a period less than 10 milliseconds?")
fast = df[df["P0"] < 0.01]
print(f"  Answer: {len(fast)}")


# --- Q3: Globular cluster pulsars ---
print("\nQ3: How many pulsars are in globular clusters?")
gc = df[df["ASSOC"].str.contains("GC", case=False, na=False)]
print(f"  Answer: {len(gc)}")


# --- Q4: Binary pulsars ---
print("\nQ4: How many binary pulsars are known?")
binary = df[df["PB"].notna()]
print(f"  Answer: {len(binary)}")


# --- Q5: Crab pulsar properties ---
print("\nQ5: What is the period and period derivative of the Crab pulsar?")
crab = df[df["JNAME"] == "J0534+2200"]
if len(crab) == 0:
    crab = df[df["PSRJ"].str.contains("0534", na=False)]
if len(crab) > 0:
    row = crab.iloc[0]
    print(f"  P0 = {row['P0']:.10f} s")
    print(f"  P1 = {row['P1']:.4e}")
else:
    print("  Crab pulsar not found!")


# --- Q6: Specific pulsar DM ---
print("\nQ6: What is the DM of PSR J0437-4715?")
j0437 = df[df["JNAME"] == "J0437-4715"]
if len(j0437) > 0:
    print(f"  DM = {j0437.iloc[0]['DM']:.4f} pc cm^-3")
else:
    print("  PSR J0437-4715 not found!")


# --- Q7: Multi-parameter query ---
print("\nQ7: How many pulsars have both a measured P1 and DM > 100?")
p1_and_dm = df[(df["P1"].notna()) & (df["DM"].notna()) & (df["DM"] > 100)]
print(f"  Answer: {len(p1_and_dm)}")


# --- Q8: Median spin period ---
print("\nQ8: What is the median spin period of all pulsars?")
median_p0 = df["P0"].median()
print(f"  Median P0 = {median_p0:.6f} s ({median_p0 * 1000:.3f} ms)")


# --- Q9: Mean DM in the Galactic plane ---
print("\nQ9: What is the mean DM of pulsars in the Galactic plane (|GB| < 5 degrees)?")
galactic_plane = df[(df["GB"].notna()) & (df["GB"].abs() < 5)]
mean_dm = galactic_plane["DM"].mean()
print(f"  Mean DM = {mean_dm:.2f} pc cm^-3 (from {len(galactic_plane)} pulsars)")


# --- Q10: Characteristic age of Crab ---
print("\nQ10: What is the characteristic age of the Crab pulsar?")
if len(crab) > 0:
    row = crab.iloc[0]
    p0, p1 = row["P0"], row["P1"]
    if p0 > 0 and p1 > 0:
        age_s = p0 / (2 * p1)
        age_yr = age_s / (365.25 * 24 * 3600)
        print(f"  Characteristic age = {age_yr:.1f} years")


# --- Q11: Magnetars (BSURF > 10^14 G) ---
print("\nQ11: How many pulsars have a surface magnetic field above 10^14 Gauss?")
has_p0_p1 = df[(df["P0"].notna()) & (df["P1"].notna()) & (df["P1"] > 0)]
bsurf = 3.2e19 * np.sqrt(has_p0_p1["P0"] * has_p0_p1["P1"])
magnetars = bsurf[bsurf > 1e14]
print(f"  Answer: {len(magnetars)} (computed from P0 and P1)")


print("\n" + "=" * 70)
print("Done. These values can be used to verify ATNF-Chat responses.")
