import mujoco
import mujoco.viewer
import time
import numpy as np
import utils

# Загрузка модели
model_path = "/home/mk/Desktop/MuJoCo/ant.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

target_pos = np.array([0, 0, 0.55]) #координаты z: 0.55 - defoult; 0.61 - hight; 0.36 - low
task_id = 0
id_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "red_torso_geom")

# Параметры ПД-регулятора
kp = 1000.0  # Коэффициент пропорциональности (жесткость)
kd = 100.0   # Коэффициент дифференциальный (демпфирование)

#def my_control(model, data):
#    global target_pos, id_geom, task_id
#    curr_targ_pos = target_pos[task_id]

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Нажмите Ctrl+C для выхода")
    
    utils.look_at_xz(viewer)
    
    #mujoco.set_mjcb_control(my_best_control)  #нужно для синхронизации текущего положения, чтоб моя функция работала с уже имеющимися данными, а не перезаписывала на новые. в общем точка отсчета как я понял
    
    # Симуляция на несколько секунд
    start_time = time.time()
    while time.time() - start_time < 1000:  # 5 секунд
        step_start = time.time()
        current_time = step_start - start_time
        
        # 2. РАСЧЕТ ОШИБКИ (это и есть "задача" для IK)
        # Получаем текущее положение и ориентацию эффектора
        geom_pos = data.geom(id_geom).xpos
        
        # Вектор ошибки по положению
        pos_error = target_pos - geom_pos
        
        # 3. РАСЧЕТ УПРАВЛЯЮЩЕГО СИГНАЛА (Якобиан и ПД-закон)
        # Вычисляем якобиан для конечного эффектора (матрица 3x nv)
        jac_pos = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(model, data, jac_pos, None, id_geom)
        
        # Расчет сил: J^T * (Kp * error - Kd * (J * qvel))
        # Это стандартная форма ПД-регулятора в пространстве задач (operational space)
        p_term = kp * pos_error
        d_term = kd * jac_pos @ data.qvel
        forces = jac_pos.T @ (p_term - d_term)
        
        # 4. ПРИМЕНЯЕМ РАСЧИТАННЫЕ СИЛЫ К СУСТАВАМ
        data.qfrc_applied = forces
        
        # Выполняем шаг симуляции
        mujoco.mj_step(model, data) 
        
        # Синхронизируем визуализацию
        viewer.sync()
        
        # Задержка для реального времени
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("Симуляция завершена")