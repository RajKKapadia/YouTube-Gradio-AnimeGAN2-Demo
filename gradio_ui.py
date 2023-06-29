import gradio as gr
import torch

face2paint = torch.hub.load(
    source='local',
    repo_or_dir='model/AK391-animegan2-pytorch-50b4c15',
    model='face2paint', size=512, side_by_side=False)

paprika = torch.hub.load(
    source='local',
    repo_or_dir='model/AK391-animegan2-pytorch-50b4c15',
    model='generator', pretrained='model/paprika.pt')

celeba_distill = torch.hub.load(
    source='local',
    repo_or_dir='model/AK391-animegan2-pytorch-50b4c15',
    model='generator', pretrained='model/celeba_distill.pt')

facepaint_v1 = torch.hub.load(
    source='local',
    repo_or_dir='model/AK391-animegan2-pytorch-50b4c15',
    model='generator', pretrained='model/face_paint_512_v1.pt')

facepaint_v2 = torch.hub.load(
    source='local',
    repo_or_dir='model/AK391-animegan2-pytorch-50b4c15',
    model='generator', pretrained='model/face_paint_512_v2.pt')


def inference(img, ver):
    if ver == 'Facepaint V1':
        out = face2paint(facepaint_v1, img)
    elif ver == 'Facepaint V2':
        out = face2paint(facepaint_v2, img)
    elif ver == 'Paprika':
        out = face2paint(paprika, img)
    else:
        out = face2paint(celeba_distill, img)
    return out


title = "AnimeGANv2"
description = "Demo for AnimeGanv2 Face Portrait. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Please use a cropped portrait picture for best results similar to the examples below."
article = "[GitHub Repository]()"
examples = [['sample/Raj_Photo.JPG', 'Facepaint V1'], ['sample/Raj_Photo.JPG', 'Facepaint V2'],
            ['sample/Raj_Photo.JPG', 'Paprika'], ['sample/Raj_Photo.JPG', 'Celeba Distill']]

demo = gr.Interface(
    fn=inference,
    inputs=[gr.components.Image(type="pil", width=512, height=512), gr.components.Radio(
        ['Facepaint V1', 'Facepaint V2', 'Paprika', 'Celeba Distill'], type="value", label='version', value='Facepaint V2')],
    outputs=gr.components.Image(type="pil", width=512, height=512),
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging='never')

demo.launch()
