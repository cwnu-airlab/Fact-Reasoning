import re 

def extract_integer_numbers(string):
    pattern = r'\d+'  # Regular expression pattern to match one or more digits
    numbers = re.findall(pattern, string)
    return [int(num) for num in numbers]

# This code is taken from http://dr-roach.com/blog/korean-postposition/
def ends_with_jong(kstr):
    try:
        m = re.search("[가-힣]+", kstr)
    except TypeError:
        return False
    if m:
        k = m.group()[-1]
        return (ord(k) - ord("가")) % 28 > 0
    else:
        return False

# This code is taken from http://dr-roach.com/blog/korean-postposition/
def en(kstr):
    josa = "은" if ends_with_jong(kstr) else "는"
    return josa

def replace_str(original_str, index, character):
    s = list(original_str)
    s[index] = character
    return "".join(s)