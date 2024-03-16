import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import gc

from biggan import BigGAN
from gan import GAN
from upscalers import get_real_esrgan
from remove_bg import get_remove_bg_model, remove_bg
from item import UnsavedItem, Item
import util
from anims import Anim
from pipeline import generate_from_config

gan = BigGAN(256)
upsampler = get_real_esrgan().cpu()
session = get_remove_bg_model()

unsaved_items = []
saved_items = [Item(i) for i in util.get_saved_item_ids()]

gen_idx = util.get_next_unsaved_idx()
selected_idx = None

with (gr.Blocks() as demo):
    with gr.Tab("Gen") as gentab:
        gr.Markdown("""
        # G e n e r a t e
        """)

        with gr.Row():
            # Settings + save gallery
            with gr.Column():
                # Settings
                with gr.Group():
                    with gr.Column(scale=1):
                        with gr.Row():
                            classes = gr.Textbox(label="Classes", show_label=True, info="(comma separated list)")
                            trunc = gr.Slider(0, 1, label="Truncation", show_label=True, value=0.5)
                        with gr.Row():
                            zmul = gr.Slider(0, 10, step=0.1, label="z-mul", show_label=True, value=1.)
                            cmul = gr.Slider(0, 10, step=0.1, label="c-mul", show_label=True, value=1.)
                            use_upscale = gr.Checkbox(value=False, label="Upscale", info="RealESRGAN-4x")
                            use_bg_remove = gr.Checkbox(value=False, label="BG Remove")
                        with gr.Row():
                            batch_size = gr.Slider(1, 32, step=1, label="Batch size", show_label=True, value=4)
                            reps = gr.Slider(1, 32, step=1, label="Repeats", show_label=True, value=1)
                        gen_btn = gr.Button("Generate", variant="primary")

                saved_gallery = gr.Gallery([i.img_path for i in saved_items],
                                           interactive=False,
                                           columns=6,
                                           scale=1,
                                           label="Saved stuff")

            # Gen gallery + save button
            with gr.Column(scale=1):
                save_btn = gr.Button("Save Selected", variant="primary")
                gen_gallery = gr.Gallery([],
                                         interactive=False,
                                         columns=6,
                                         scale=1,
                                         label="Generations")

        @gen_btn.click(inputs=[trunc, zmul, cmul, classes, batch_size, reps, use_upscale, use_bg_remove],
                       outputs=gen_gallery)
        def letsgo(trunc, zmul, cmul, classes, batch_size, reps, use_upscale, use_bg_remove, progress=gr.Progress()):
            global image_list, gen_idx, selected_idx

            ctr = 0
            progress((0, reps * batch_size))
            last_desc = ""

            def ctr_callback(x=None):
                nonlocal ctr, last_desc
                if isinstance(x, str):
                    last_desc = x
                else:
                    if x is None:
                        ctr += 1
                    else:
                        ctr += x
                progress((ctr, reps * batch_size), desc=last_desc)

            selected_idx = None

            config = {
                "trunc": trunc,
                "zmul": zmul,
                "cmul": cmul,
                "upscale": use_upscale,
                "bg_remove": use_bg_remove,
                "gan": "BigGAN"
            }

            batch_size = int(batch_size)
            reps = int(reps)

            for _ in progress.tqdm(range(reps)):
                progress((ctr, reps * batch_size), desc="GAN running")

                zs = gan.get_z(batch_size, classes=[i for i in classes.split(",") if len(i) > 0], z_mul=zmul,
                               class_mul=cmul, trunc=trunc)

                imgs = generate_from_config(config, zs, gan, upsampler, session, batch_size, ctr_callback)

                for i in range(batch_size):
                    uitem = UnsavedItem(gen_idx, zs[i], imgs[i], config)
                    unsaved_items.insert(0, uitem)
                    gen_idx += 1

            return [i.img_path for i in unsaved_items]

        @gen_gallery.select()
        def what(evt: gr.SelectData):
            global selected_idx
            selected_idx = evt.index

        @save_btn.click(outputs=saved_gallery)
        def save_clicked():
            global saved_items, selected_idx

            if selected_idx is None:
                print("aint nothing selected")
            else:
                item = unsaved_items[selected_idx].to_item()
                saved_items.insert(0, item)
            selected_idx = None
            return [i.img_path for i in saved_items]

    with gr.Tab("Anim") as animtab:
        gr.Markdown("""
        # A n i m a t e
        """)

        anims = {}
        anim_item = None

        with gr.Row():
            with gr.Column():
                anim_prev = gr.Image(interactive=False)

            # Settings
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        min_cycles = gr.Slider(1, 10, step=1, label="Min cycles", show_label=True, value=1)
                        max_cycles = gr.Slider(1, 20, step=1, label="Max cycles", show_label=True, value=3)

                    with gr.Row():
                        amplitude = gr.Slider(0, 1, label="amt", value=0.1)
                        batch_size2 = gr.Slider(1, 32, step=1, label="Batch size", show_label=True, value=4)

                    with gr.Row():
                        # gr.Slider(1, 100, step=1, label="Num frames", show_label=True, value=24)
                        nframes = gr.Number(value=24, precision=0, minimum=0, step=1)
                        fps = gr.Slider(1, 40, step=1, label="FPS", show_label=True, value=24)

                    make_anim = gr.Button("Animate", variant="primary")

                with gr.Group():
                    anim_name = gr.Textbox(value="Name", interactive=True)
                    save_anim = gr.Button("Save anim", variant='primary')

            # Loadable animations
            anim_name_list = gr.Dataset(components=["Textbox"], samples=[], type="values", label="Anims")

        saved_gallery2 = gr.Gallery([i.img_path for i in saved_items],
                                    interactive=False,
                                    columns=6,
                                    scale=1,
                                    label="Saved stuff",
                                    preview=False,
                                    allow_preview=False)

        tmp_anim = None

        @save_anim.click(inputs=[anim_name], outputs=[anim_name, anim_name_list])
        def save_anim_b(anim_name):
            global tmp_anim

            if anim_item is None:
                print("select item first silly")
                return

            if tmp_anim is None:
                print("no anim to save")

            if os.path.exists(os.path.join(anim_item.root, anim_name)):
                print("But that anim already exists")
                return

            tmp_anim.save_with_name(anim_name)
            anims[tmp_anim.name] = tmp_anim

            tmp_anim = None
            return "", [[i] for i in sorted(anims)]

        @make_anim.click(inputs=[batch_size2, nframes, amplitude, min_cycles, max_cycles, fps], outputs=[anim_prev])
        def mk_anim(batch_size2, nframes, amplitude, min_cycles, max_cycles, fps, progress=gr.Progress()):
            global tmp_anim
            if anim_item is None:
                print("select item first silly")
                return

            ctr = 0
            progress((0, nframes))
            last_desc = ""

            def ctr_callback(x=None):
                nonlocal ctr, last_desc
                if isinstance(x, str):
                    last_desc = x
                else:
                    if x is None:
                        ctr += 1
                    else:
                        ctr += x
                progress((ctr, nframes), desc=last_desc)

            anim = Anim.sin_walk(anim_item, steps=nframes, amplitude=amplitude, min_cycles=min_cycles,
                                 max_cycles=max_cycles)
            anim.save_images_and_make_gif(gan, upsampler, session, batch_size=batch_size2, fps=fps, progress=ctr_callback)
            tmp_anim = anim

            return anim.gif_path

        @saved_gallery2.select(outputs=[anim_prev, anim_name_list])
        def select_item_to_animate(data: gr.SelectData):
            global anims, anim_item, tmp_anim
            tmp_anim = None
            anim_item = saved_items[data.index]
            names = util.get_anim_names(anim_item.idx)
            anims = {name: Anim(anim_item.root, anim_item.config, name) for name in names}

            return anim_item.img_path, [[i] for i in sorted(anims)]

        @anim_name_list.select(outputs=[anim_prev])
        def select_anim_name(data: gr.SelectData):
            name = data.value[0]
            anim = anims[name]
            return anim.gif_path

        @animtab.select(outputs=saved_gallery2)
        def refresh(data: gr.SelectData):
            global tmp_anim
            tmp_anim = None
            return [i.img_path for i in saved_items]

demo.launch(share=True)
