import os

def beep_alert():
    try:
        import winsound
        winsound.Beep(1000, 300)
    except:
        os.system('echo "\a"')