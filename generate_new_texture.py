import bpy

def generate_uv_coordinates(input_path, output_path):
    # 清空场景
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 导入模型
    bpy.ops.import_scene.obj(filepath=input_path)

    # 获取导入的对象
    obj = bpy.context.selected_objects[0]

    # 选择对象并进入编辑模式
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # 生成 UV 坐标（智能 UV 投影）
    bpy.ops.uv.smart_project()

    # 返回对象模式
    bpy.ops.object.mode_set(mode='OBJECT')

    # 导出带有 UV 坐标的模型
    bpy.ops.export_scene.obj(filepath=output_path)

# 输入和输出模型路径
input_model_path = '/home/zcb/self_code_training/InTeX_self/data/9-opt.obj'
output_model_path = '/home/zcb/self_code_training/InTeX_self/data/9-opt_new.obj'

# 生成 UV 坐标并导出模型
generate_uv_coordinates(input_model_path, output_model_path)