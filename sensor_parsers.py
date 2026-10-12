import re

#Example:
#F1=270 F2=180 F3=156 F4=174 F5=165
flex_regex= re.compile(
    r"F1=([-0-9.]+)\s+"
    r"F2=([-0-9.]+)\s+"
    r"F3=([-0-9.]+)\s+"
    r"F4=([-0-9.]+)\s+"
    r"F5=([-0-9.]+)"
)

#Example:
#GYRx=-0.23 GYRy=4.758 GYRz=1.926
#ACCx_body=0.123 ACCy_body=0.456 ACCz_body=0.789
gyro_regex= re.compile(
    r"GYRx=([-0-9.]+)\s+"
    r"GYRy=([-0-9.]+)\s+"
    r"GYRz=([-0-9.]+)"
)

acc_body_regex= re.compile(
    r"ACCx_body=([-0-9.]+)\s+"
    r"ACCy_body=([-0-9.]+)\s+"
    r"ACCz_body=([-0-9.]+)"
)

def parse_flex(line):
    match= flex_regex.search(line)
    if not match:
        return None
    try:
        return list(map(float, match.groups()))
    except ValueError:
        return None

def parse_imu(line):
    gyro= gyro_regex.search(line)
    acc_body= acc_body_regex.search(line)

    if not (gyro and acc_body):
        return None

    try:
        gyro_values= list(map(float, gyro.groups()))
        acc_values= list(map(float, acc_body.groups()))

        return {
            "gyro": gyro_values,
            "acc_body": acc_values
        }
    except ValueError:
        return None