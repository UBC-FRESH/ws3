import time

def timed(func):
    def wrapper(*args):
        t = time.time()
        result = func(*args)
        t = time.time() - t
        print '%s took %.3f seconds.' % (func.func_name, t)
        return result
    return wrapper

HORIZON_DEFAULT = 30
PERIOD_LENGTH_DEFAULT = 5
MIN_AGE_DEFAULT = 0
MAX_AGE_DEFAULT = 100
CURVE_EPSILON_DEFAULT = 0.01
AREA_EPSILON_DEFAULT = 0.01
SPECIES_GROUPS_QC  = {
    'ERR':'ERR',
    'ERS':'ERS',
    'BOP':'BOP',
    'EPR':'SEP',
    'CHB':'FTO',
    'EPN':'SEP',
    'EPO':'SEP',
    'BOJ':'BOJ',
    'PEH':'PEU',
    'ERA':'ERR',
    'CAC':'FTO',
    'ERN':'ERR',
    'PEG':'PEU',
    'EPB':'SEP',
    'CAF':'FTO',
    'PEB':'PEU',
    'BOG':'BOP',
    'SOA':'NCO',
    'SAL':'NCO',
    'SAB':'SAB',
    'PIB':'PIN',
    'PIG':'SEP',
    'PRU':'AUR',
    'PET':'PEU',
    'CET':'FTO',
    'PRP':'NCO',
    'PIR':'PIN',
    'PIS':'SEP',
    'PED':'PEU',
    'FRA':'FTO',
    'CHE':'FTO',
    'CHG':'FTO',
    'FRN':'FTO',
    'THO':'AUR',
    'CHR':'FTO',
    'FRP':'FTO',
    'TIL':'FTO',
    'MEL':'AUR',
    'ORT':'FTO',
    'ORR':'FTO',
    'MEH':'AUR',
    'NOC':'FTO',
    'HEG':'HEG',
    'OSV':'FTO',
    'ORA':'FTO'
}

SPECIES_GROUPS_WOODSTOCK_QC  = {
    'ERR':'ERR',
    'ERS':'ERS',
    'BOP':'BOP',
    'EPR':'SEP',
    'CHB':'FTO',
    'EPN':'SEP',
    'EPO':'SEP',
    'BOJ':'BOJ',
    'PEH':'PEU',
    'ERA':'ERR',
    'CAC':'FTO',
    'ERN':'ERR',
    'PEG':'PEU',
    'EPB':'SEP',
    'CAF':'FTO',
    'PEB':'PEU',
    'BOG':'BOP',
    'SOA':'NCO',
    'SAL':'NCO',
    'SAB':'SAB',
    'PIB':'PIN',
    'PIG':'SEP',
    'PRU':'AUR',
    'PET':'PEU',
    'CET':'FTO',
    'PRP':'NCO',
    'PIR':'PIN',
    'PIS':'SEP',
    'PED':'PEU',
    'FRA':'FTO',
    'CHE':'FTO',
    'CHG':'FTO',
    'FRN':'FTO',
    'THO':'AUR',
    'CHR':'FTO',
    'FRP':'FTO',
    'TIL':'FTO',
    'MEL':'AUR',
    'ORT':'FTO',
    'ORR':'FTO',
    'MEH':'AUR',
    'NOC':'FTO',
    'HEG':'HEG',
    'OSV':'FTO',
    'ORA':'FTO'
}

##########################################
# keys correspond to bin labels
# values correspond to bin upper bounds (inclusive)
AGE_CLASS_BINS_DEFAULT = {
    '10':20,
    '30':40,
    '50':60,
    '70':80,
    '90':100,
    '120+':MAX_AGE_DEFAULT
}
##########################################
    

def is_num(s):
    try:
        float(s)
        return True
    except:
        return False


