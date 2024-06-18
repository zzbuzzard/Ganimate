import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import gc

from biggan import BigGAN
from stylegan import StyleGAN
from gan import GAN
from upscalers import get_real_esrgan
from remove_bg import get_remove_bg_model, remove_bg
from item import UnsavedItem, Item
import util
from anims import Anim
from pipeline import generate_from_config


def gan_from_name(name: str) -> GAN:
    if name == "BigGAN":
        return BigGAN(256)
    elif name == "BigGAN-512":
        return BigGAN(512)
    elif name == "StyleGAN-ffhq":
        return StyleGAN("ffhq")
    elif name == "StyleGAN-cat":
        return StyleGAN("cat")
    elif name == "StyleGAN-human":
        return StyleGAN("human")
    else:
        raise NotImplementedError(f"Unknown GAN: '{name}'")


gan = gan_from_name("BigGAN")
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
                        # Per-GAN stuff
                        with gr.Tab("BigGAN") as biggantab:
                            with gr.Row():
                                bg_res = gr.Checkbox(value=False, label="512x512")
                                # info="(comma separated list)"
                                classes = gr.Textbox(label="Classes (comma sep)", show_label=True)
                                cmul = gr.Slider(0, 10, step=0.1, label="c-mul", show_label=True, value=1.)
                        with gr.Tab("StyleGAN") as stylegantab:
                            with gr.Row():
                                sg_mode = gr.Dropdown(choices=["ffhq", "cat", "human"], value="ffhq", show_label=False,
                                                   interactive=True, allow_custom_value=False)
                                wmul = gr.Slider(0, 10, step=0.1, label="w-mul", show_label=True, value=1.)
                                wnoise = gr.Slider(0, 10, step=0.1, label="w-noise", show_label=True, value=0.)
                        with gr.Row():
                            zmul = gr.Slider(0, 10, step=0.1, label="z-mul", show_label=True, value=1.)
                            trunc = gr.Slider(0, 1, label="Truncation", show_label=True, value=0.5)
                            use_upscale = gr.Checkbox(value=False, label="Upscale", info="RealESRGAN-4x")
                            use_bg_remove = gr.Checkbox(value=False, label="BG Remove")
                        with gr.Row():
                            batch_size = gr.Slider(1, 32, step=1, label="Batch size", show_label=True, value=4)
                            reps = gr.Slider(1, 32, step=1, label="Repeats", show_label=True, value=1)
                        with gr.Row():
                            use_xmirror = gr.Checkbox(value=False, label="X-Tile")
                            use_ymirror = gr.Checkbox(value=False, label="Y-Tile")
                            # TODO: more settingseee
                        with gr.Row():
                            tag = gr.Textbox(label="Tag?", show_label=True)

                        gen_btn = gr.Button("Generate", variant="primary")

                        # TODO: dont forget to add new settings here!
                        gen_settings = {classes, cmul, wmul, wnoise, zmul, trunc, use_upscale, use_bg_remove, batch_size, reps, use_xmirror, use_ymirror, tag}


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

        def config_from_settings(data):
            # The config only really needs to contain stuff needed for animation; this is a bit excessive atm (eg wmul)
            return {
                "trunc": data[trunc],
                "zmul": data[zmul],
                "cmul": data[cmul],
                "wmul": data[wmul],
                "wnoise": data[wnoise],
                "upscale": data[use_upscale],
                "bg_remove": data[use_bg_remove],
                "x_mirror": data[use_xmirror],
                "y_mirror": data[use_ymirror],
                "tag": data[tag],
                "gan": gan.name,
            }

        def gen_kwargs_from_settings(data):
            return {
                "trunc": data[trunc],
                "z_mul": data[zmul],
                "class_mul": data[cmul],
                "w_mul": data[wmul],
                "wplus_noise": data[wnoise],
                "classes": [i.strip() for i in data[classes].strip().split(",") if len(i) > 0]
            }

        SELECTED_MODEL = "BigGAN"
        # Track current model
        def set_bg_mode(is_512: bool):
            global SELECTED_MODEL
            SELECTED_MODEL = "BigGAN" + ("-512" if is_512 else "")
            print("Selected", SELECTED_MODEL)
        def set_sg_mode(mode: str):
            global SELECTED_MODEL
            SELECTED_MODEL = f"StyleGAN-{mode}"
            print("Selected", SELECTED_MODEL)
        biggantab.select(set_bg_mode, inputs=bg_res)
        bg_res.input(set_bg_mode, inputs=bg_res)
        stylegantab.select(set_sg_mode, inputs=sg_mode)
        sg_mode.input(set_sg_mode, inputs=sg_mode)

        # Updates the current GAN to that specified by SELECTED_MODEL
        def update_gan(update_to=None):
            global gan
            goal = SELECTED_MODEL if update_to is None else update_to
            if gan.name != goal:
                print(f"Switching GAN: {gan.name} -> {goal}")
                del gan
                gc.collect()
                torch.cuda.empty_cache()
                gan = gan_from_name(goal)
                assert gan.name == goal, f"{gan.name} != {goal} ?!"

        @gen_btn.click(inputs=gen_settings, outputs=gen_gallery)
        def letsgo(data, progress=gr.Progress()):
            global image_list, gen_idx, selected_idx

            update_gan()

            bs = data[batch_size]
            rep = data[reps]

            ctr = 0
            progress((0, rep * bs))
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
                progress((ctr, rep * bs), desc=last_desc)

            selected_idx = None

            config = config_from_settings(data)

            bs = int(bs)
            rep = int(rep)
            kwargs = gen_kwargs_from_settings(data)

            for _ in progress.tqdm(range(rep)):
                progress((ctr, rep * bs), desc="GAN running")

                zs = gan.get_z(bs, **kwargs)
                imgs = generate_from_config(config, zs, gan, upsampler, session, bs, ctr_callback)

                for i in range(bs):
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

            need_gan = anim_item.config["gan"]
            update_gan(need_gan)

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
