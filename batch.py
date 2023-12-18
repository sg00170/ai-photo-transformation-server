import os
import time
from time import strftime

def log(str = ''):
    return print(f"[{strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {str}")

log("AI 포토존 프로그램 시작 준비...")

log("AI 포토존 프로그램 서버 시작")
os.system("httpd.exe -k restart")
log("AI 포토존 프로그램 서버 시작 완료")

log("AI 포토존 프로그램 DB 시작")
os.system("net start mysql80")
log("AI 포토존 프로그램 DB 시작 완료")

log("AI 포토존 프로그램 시작")
os.system("start /d \"C:\\Users\\misun\\PycharmProjects\\nauthybomb\\dist\\main\\\" /b main.exe")
log("AI 포토존 프로그램 시작 완료")

log("AI 포토존 프로그램 준비 완료...")