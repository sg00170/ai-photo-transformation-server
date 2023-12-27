import torch
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import pymysql
import sys
import os
import time
from time import strftime
from enum import Enum
import random
from diffusers.utils import load_image
import cv2
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
import numpy as np
from PIL import Image
import math
import pysftp
import logging
from logging import handlers

# Resource
def resource_path(relative_path):
    try:
        # PyInstaller에 의해 임시폴더에서 실행될 경우 임시폴더로 접근하는 함수
        base_path = sys._MEIPASS
    except Exception:
        # base_path = os.path.abspath(".")
        base_path = ''
    return os.path.join(base_path, (relative_path[2:]).replace('/', '\\'))

load_dotenv()
scheduler = BackgroundScheduler()
logger = logging.getLogger('main')
daily_log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s(Line:%(lineno)d): %(message)s')
daliy_log_handler = handlers.TimedRotatingFileHandler(
    filename=resource_path('./logs/naughtybom.log'),
    when='midnight',
    interval=1,
    encoding='utf-8'
)
daliy_log_handler.setFormatter(daily_log_formatter)
daliy_log_handler.suffix = "%Y%m%d"
logger.setLevel(logging.INFO)
logger.addHandler(daliy_log_handler)

# DTO
user_columns = [
    'id', 'prompt_setting_id', 'phone_number', 'gender', 'request_time',
    'state', 'process_time', 'created_at', 'updated_at', 'deleted_at'
]
prompt_setting_columns = [
    'id', 'name', 'prompt', 'negative_prompt', 'model', 'lora',
    'embedding', 'step', 'strength_min', 'strength_max', 'lora_scale',
    'grade', 'created_at', 'updated_at', 'deleted_at'
]

# Enum
class UserTransformationState(Enum):
    PENDING = 0
    PROCESSING = 1
    SUCCEED = 2
    RETRYING = 10
    FAILED = 20

class PromptSettingGrade(Enum):
    SSR = 0
    SR = 1
    S = 2
    R = 3
    A = 4
    B = 5
    C = 7
    D = 8
    E = 9
    F = 10

def log(str = '', level = logging.INFO):
    if (level == logging.INFO):
        logger.info(str)
    elif (level == logging.DEBUG):
        logger.debug(str)
    else :
        logger.error(str)
    print(f"[{strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {str}")

def set_strength_range(min = 0.5, max = 0.5):
    ranges = []
    while(min <= max):
        ranges.append(min)
        min += 0.1
    return ranges

# 스케쥴러에서 실행될 함수
def process(mod = 0):
    try:
        connect = pymysql.connect(
            host=os.environ.get('DB_HOST'),
            user=os.environ.get('DB_USERNAME'),
            password=os.environ.get('DB_PASSWORD'),
            db=os.environ.get('DB_DATABASE'),
            charset='utf8'
        )
        cursor = connect.cursor()
        # cursor.execute(
        #     'select * from users where state = %s and MOD(id, 2) = %s order by id asc limit 1 for update',
        #     (UserTransformationState.PENDING.value, int(mod))
        # )
        cursor.execute(
            'select * from users where state = %s order by id asc limit 1 for update',
            (UserTransformationState.PENDING.value)
        )
        user = cursor.fetchone()
        if (user != None):
            user_id = user[user_columns.index('id')]
            log(f"<MOD{mod}> {user_id} - {user[user_columns.index('phone_number')]} 진행 =====")
            # 대기 인원
            cursor.execute(
                'select count(*) from users where state = %s and MOD(id, 2) = %s and id != %s',
                (UserTransformationState.PENDING.value, mod, user_id)
            )
            log(f"<MOD{mod}> 앞으로 대기 {cursor.fetchone()[0]}명...")
            cursor.execute('select * from prompt_settings')
            # 프롬프트 선정
            prompt_settings = cursor.fetchall()
            prompt_setting = random.choice(prompt_settings)
            log(f"<MOD{mod}> Prompt: {prompt_setting[prompt_setting_columns.index('name')]} 선정 >> {PromptSettingGrade(prompt_setting[prompt_setting_columns.index('grade')]).name}")
            # 유저 상태 변경
            cursor.execute(
                'update users set state = %s, prompt_setting_id = %s where id = %s',
                (UserTransformationState.PROCESSING.value, prompt_setting[prompt_setting_columns.index('id')], user_id)
            )
            connect.commit()
            
            # 이미지 로드 및 변환
            try:
                filename = f"{user[user_columns.index('request_time')]}_naughtybomb"
                origin_image = load_image(f"{os.environ.get('ORIGIN_URL')}{filename}.png")
                origin_image.save(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\origin\\{filename}.png")
                image = np.array(origin_image)
                origin_image = origin_image.convert("L")
                low_threshold = 100
                high_threshold = 200
                image = cv2.Canny(image, low_threshold, high_threshold)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                control_image = Image.fromarray(image)
            except Exception as e:
                cursor.execute(
                    'update users set state = %s, where id = %s',
                    (UserTransformationState.PENDING.value, user_id)
                )
                connect.commit()
                connect.close()
                log(f"image exception 발생...", logging.ERROR)
                log(e)
                
            # 컨트롤러 및 모델 셋팅
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                resource_path("./models/" + prompt_setting[prompt_setting_columns.index('model')] + ".safetensors"),
                controlnet=controlnet,
                torch_dtype=torch.float16
            ).to('cuda')
            # 스케쥴러 셋팅
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)  # DPM++ 2M # OK(20s)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                {"algorithm_type": "sde-dpmsolver++"})  # DPM++ 2M SDE
            # LoRa 셋팅
            pipe.load_lora_weights(resource_path("./LoRa/more_details.safetensors"))
            if (prompt_setting[prompt_setting_columns.index('lora')] != None):
                pipe.load_lora_weights(
                    resource_path("./LoRa/" + prompt_setting[prompt_setting_columns.index('lora')] + ".safetensors"))
            # embedding 셋팅
            pipe.load_textual_inversion(resource_path("./embeddings/style-rustmagic.pt"), token="style-rustmagic")
            pipe.load_textual_inversion(resource_path("./embeddings/UnrealisticDream.pt"), token="UnrealisticDream")
            pipe.load_textual_inversion(resource_path("./embeddings/DarkFantasy.pt"), token="DarkFantasy")
            pipe.enable_model_cpu_offload()
            # 이미지 변환
            generator = torch.manual_seed(0)
            step = prompt_setting[prompt_setting_columns.index('step')]
            start = time.time()
            math.factorial(100000)
            strength_ranges = set_strength_range(
                prompt_setting[prompt_setting_columns.index('strength_min')],
                prompt_setting[prompt_setting_columns.index('strength_max')]
            )
            strength = random.choice(strength_ranges)
            image = pipe(
                prompt=prompt_setting[prompt_setting_columns.index('prompt')],
                negative_prompt=prompt_setting[prompt_setting_columns.index('negative_prompt')],
                num_inference_steps=step,
                generator=generator,
                image=origin_image,
                control_image=control_image,
                guidance_scale=7,
                strength=strength,
                cross_attention_kwargs={"scale": prompt_setting[prompt_setting_columns.index('lora_scale')]}
            ).images[0]
            end = time.time()
            log(f"<MOD{mod}> 이미지 변환 {end - start:.5f} 소요 / Strength: {strength} / step: {step} / lora: {prompt_setting[prompt_setting_columns.index('lora_scale')]}")
            # 이미지 저장
            image.save(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\transformation\\{filename}.png")
            #워터마크 및 영상 저장
            watermark_array = np.fromfile(resource_path('./assets/watermark.png'), np.uint8)
            # watermark = cv2.imread(resource_path('./assets/watermark.png'), cv2.IMREAD_UNCHANGED)
            watermark = cv2.imdecode(watermark_array, cv2.IMREAD_UNCHANGED)
            origin = cv2.imread(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\origin\\{filename}.png")
            transformation = cv2.imread(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\transformation\\{filename}.png")
            _, mask = cv2.threshold(watermark[:,:,3], 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            watermark = cv2.cvtColor(watermark, cv2.COLOR_BGRA2BGR)
            h, w = watermark.shape[:2]

            masked_images = []
            masked_video_frames = []
            fps = 30
            for name, image in {'origin': origin, 'transformation': transformation}.items():
                image_h, image_w = image.shape[:2]
                roi = image[image_h-h-20:image_h-20, image_w-w-20:image_w-20]
                masked_watermark = cv2.bitwise_and(watermark, watermark, mask=mask)
                masked_image = cv2.bitwise_and(roi, roi, mask=mask_inv)
                combined = masked_watermark + masked_image
                image[image_h-h-20:image_h-20, image_w-w-20:image_w-20] = combined
                masked_images.append(image)
                # 워터마크 이미지 저장
                cv2.imwrite(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\{name}\\{filename}.png", image)
                # 이미지 원격
                cnOptions = pysftp.CnOpts()
                try: 
                    with pysftp.Connection(
                        host=os.environ.get('SFTP_HOST'),
                        username=os.environ.get('SFTP_USERNAME'),
                        private_key=resource_path(os.environ.get('SFTP_KEY')),
                        cnopts=cnOptions,
                    ) as sftp:
                        sftp.put(
                            f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\{name}\\{filename}.png", 
                            preserve_mtime=True,
                            remotepath=f"/var/www/server/storage/app/public/{name}/{filename}.png"
                        )
                        sftp.close()
                except Exception as e:
                    log(f"pysftp exception image 발생...", logging.ERROR)
                    log(e)
                # 영상용 이미지 생성
                count = 1
                while(count <= (fps * 2)):
                    masked_video_frames.append(image)
                    count += 1
            # 디졸빙용 영상 이미지 생성
            count = 1
            while(count <= fps):
                alpha = count / fps
                frame = cv2.addWeighted(masked_images[0], 1 - alpha, masked_images[1], alpha, 0)
                masked_video_frames.insert((fps * 2) + count - 1, frame)
                count += 1
            # 디졸빙효과 영상 생성
            output = cv2.VideoWriter(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\video\\{filename}.mp4", fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=fps, frameSize=[1080, 1920])
            for image in masked_video_frames:
                output.write(image)
            output.release()
            os.system(f"ffmpeg -i C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\video\\{filename}.mp4 -vcodec libx264 C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\video\\{filename}_2.mp4")
            # 영상 원격 업로드
            cnOptions = pysftp.CnOpts()
            try: 
                with pysftp.Connection(
                    host=os.environ.get('SFTP_HOST'),
                    username=os.environ.get('SFTP_USERNAME'),
                    private_key=resource_path(os.environ.get('SFTP_KEY')),
                    cnopts=cnOptions,
                ) as sftp:
                    sftp.put(
                        f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\video\\{filename}_2.mp4", 
                        preserve_mtime=True,
                        remotepath=f"/var/www/server/storage/app/public/video/{filename}.mp4"
                    )
                    sftp.close()
            except Exception as e:
                log(f"pysftp exception video 발생...", logging.ERROR)
                log(e)
            
            # 프로세스 완료 처리
            cursor.execute(
                'update users set state = %s, process_time = %s where id = %s',
                (UserTransformationState.SUCCEED.value, int(end - start), user_id)
            )
            connect.commit()
            connect.close()
            
            # 생성된 이미지 & 영상 삭제
            os.remove(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\origin\\{filename}.png")
            os.remove(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\transformation\\{filename}.png")
            os.remove(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\video\\{filename}.mp4")
            os.remove(f"C:\\Apache24\\htdocs\\naughtybomb-web\\public\\images\\video\\{filename}_2.mp4")
            
            log(f"<MOD{mod}> 완료 ====================")
        else:
            log(f"<MOD{mod}> 처리할 내역이 없습니다...")
    except Exception as e:
        connect.rollback()
        log(f"<MOD{mod}> exception 발생...", logging.ERROR)
        log(e, logging.ERROR)

@scheduler.scheduled_job('cron', second='*/10', id='process')
def job1():
    log(f"scheduler JOB start >>>>>", logging.DEBUG)
    process()
    log(f"scheduler JOB end >>>>>", logging.DEBUG)

# @scheduler.scheduled_job('interval', seconds=10, id='process_odd')
# def job1():
#     log(f"scheduler JOB1 start >>>>>")
#     process(1)
#     log(f"scheduler JOB1 end >>>>>")
# @scheduler.scheduled_job('interval', seconds=10, id='process_even')
# def job2():
#     log(f"scheduler JOB2 start >>>>>")
#     process(0)
#     log(f"scheduler JOB2 end >>>>>")

scheduler.start()

while True:
    time.sleep(1)
    log(strftime('%Y-%m-%d %H:%M:%S', time.localtime()), logging.DEBUG)