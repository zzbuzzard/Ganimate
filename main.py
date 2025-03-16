import shutil

import gradio as gr
import torch
from PIL import Image
import numpy as np
import os
import gc
from os.path import join

import stylegan
from biggan import BigGAN
from stylegan import StyleGAN
from gan import GAN
from upscalers import get_real_esrgan
from remove_bg import get_remove_bg_model, remove_bg
from item import UnsavedItem, Item
import util
from anims import Anim, interpolate
from pipeline import generate_from_config


def gan_from_name(_name: str) -> GAN:
    if _name.endswith("_t"):
        util.set_tile_mode(True)
        name = _name
        while name.endswith("_t"):
            name = name[:-2]
    else:
        util.set_tile_mode(False)
        name = _name
    if name == "BigGAN":
        return BigGAN(256)
    elif name == "BigGAN-512":
        return BigGAN(512)
    elif name.startswith("stylegan"):
        return StyleGAN(name)
    else:
        raise NotImplementedError(f"Unknown GAN: '{_name}'")


gan = gan_from_name("BigGAN")
upsampler = get_real_esrgan().cpu()
session = get_remove_bg_model()

unsaved_items = []
saved_items = [Item(i) for i in util.get_saved_item_ids()]

gen_idx = util.get_next_unsaved_idx()
selected_idx = None

with (gr.Blocks() as demo):
    # Generate tab
    with gr.Tab("GenObj") as gentab:
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
                                sg_mode = gr.Dropdown(choices=list(stylegan.all_models), value="stylegan3-t-ffhq-1024x1024",
                                                      show_label=False, interactive=True, allow_custom_value=False)
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
                            use_tile = gr.Checkbox(value=False, label="TILE")
                            use_xmirror = gr.Checkbox(value=False, label="X-Tile (mirror)")
                            use_ymirror = gr.Checkbox(value=False, label="Y-Tile (mirror)")

                            # TODO: more settingseee
                        with gr.Row():
                            tag = gr.Textbox(label="Tag?", show_label=True)

                        gen_btn = gr.Button("Generate", variant="primary")

                        # TODO: dont forget to add new settings here!
                        gen_settings = {classes, cmul, wmul, wnoise, zmul, trunc, use_upscale, use_bg_remove, batch_size, reps, use_xmirror, use_ymirror, tag, use_tile}

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
                "gan": gan.name + ("_t" if data[use_tile] else ""),
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
            SELECTED_MODEL = mode
            print("Selected", SELECTED_MODEL)
        def set_tile_mode(tile: bool):
            global SELECTED_MODEL
            if SELECTED_MODEL[-2:] == "_t":
                if tile:
                    return
                SELECTED_MODEL = SELECTED_MODEL[:-2]
            else:
                if not tile:
                    return
                SELECTED_MODEL = SELECTED_MODEL + "_t"
        biggantab.select(set_bg_mode, inputs=bg_res)
        bg_res.input(set_bg_mode, inputs=bg_res)
        stylegantab.select(set_sg_mode, inputs=sg_mode)
        sg_mode.input(set_sg_mode, inputs=sg_mode)
        use_tile.input(set_tile_mode, inputs=use_tile)

        # Updates the current GAN to that specified by SELECTED_MODEL
        def update_gan(update_to=None, progress=None):
            global gan
            goal = SELECTED_MODEL if update_to is None else update_to
            if gan.name != goal:
                if progress is not None:
                    progress(None, desc=f"Loading GAN: {goal}")
                print(f"Switching GAN: {gan.name} -> {goal}")
                del gan
                gc.collect()
                torch.cuda.empty_cache()
                gan = gan_from_name(goal)
                if goal.endswith("_t_t"):
                    goal = goal[:-2]
                if gan.name.endswith("_t_t"):
                    gan.name=gan.name[:-2]
                assert gan.name == goal, f"{gan.name} != {goal} ?!"

        @gen_btn.click(inputs=gen_settings, outputs=gen_gallery)
        def letsgo(data, progress=gr.Progress()):
            global image_list, gen_idx, selected_idx

            update_gan(progress=progress)

            bs = data[batch_size]
            rep = data[reps]

            progress(None, desc="Loaded!")

            selected_idx = None

            config = config_from_settings(data)

            bs = int(bs)
            rep = int(rep)
            num_img = bs * rep

            kwargs = gen_kwargs_from_settings(data)

            zs = gan.get_z(num_img, **kwargs)
            imgs = generate_from_config(config, zs, gan, upsampler, session, bs, progress)

            for i in range(num_img):
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

    # Animate tab
    with gr.Tab("AnimObj") as animtab:
        gr.Markdown("""
        # A n i m a t e
        Animate the saved objects generated in GenObj. This works by just perturbing the latent slightly.
        """)

        anims = {}
        anim_item = None

        with gr.Row():
            anim_prev = gr.Image(interactive=False)
            anim_vid = gr.Video(interactive=False, autoplay=True, loop=True)

            # Settings
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        min_cycles = gr.Slider(1, 10, step=1, label="Min cycles", show_label=True, value=1)
                        max_cycles = gr.Slider(1, 20, step=1, label="Max cycles", show_label=True, value=3)

                    with gr.Row():
                        amplitude = gr.Number(0.1, label="amplitude", minimum=0, step=0.1)
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
            # textbox = gr.Textbox(interactive=False)
            anim_name_list = gr.Dataset(components=[anim_name], samples=[["test"], ["test2"]], type="values", label="Anims")

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
            return "", gr.Dataset(samples=[[i] for i in sorted(anims)])

        def checkpath(path: str):
            if not os.path.isfile(path):
                return None
            return path

        @make_anim.click(inputs=[batch_size2, nframes, amplitude, min_cycles, max_cycles, fps], outputs=[anim_prev, anim_vid])
        def mk_anim(batch_size2, nframes, amplitude, min_cycles, max_cycles, fps, progress=gr.Progress()):
            global tmp_anim
            if anim_item is None:
                print("select item first silly")
                return

            need_gan = anim_item.config["gan"]
            update_gan(need_gan, progress=progress)

            anim = Anim.sin_walk(anim_item, steps=nframes, amplitude=amplitude, min_cycles=min_cycles,
                                 max_cycles=max_cycles)
            anim.save_images_and_make_gif(gan, upsampler, session, batch_size=batch_size2, fps=fps, progress=progress)
            tmp_anim = anim

            img_path = anim_item.img_path if checkpath(anim.gif_path) is None else checkpath(anim.gif_path)

            return img_path, checkpath(anim.mp4_path)

        @saved_gallery2.select(outputs=[anim_prev, anim_vid, anim_name_list])
        def select_item_to_animate(data: gr.SelectData):
            global anims, anim_item, tmp_anim
            tmp_anim = None
            anim_item = saved_items[data.index]
            names = util.get_anim_names(anim_item.idx)
            anims = {name: Anim(anim_item.root, anim_item.config, name) for name in names}

            return anim_item.img_path, None, gr.Dataset(samples=[[i] for i in sorted(anims)])

        @anim_name_list.select(outputs=[anim_prev, anim_vid])
        def select_anim_name(data: gr.SelectData):
            name = data.value[0]
            anim = anims[name]

            img_path = anim_item.img_path if checkpath(anim.gif_path) is None else checkpath(anim.gif_path)

            return img_path, checkpath(anim.mp4_path)

        @animtab.select(outputs=saved_gallery2)
        def refresh(data: gr.SelectData):
            global tmp_anim
            tmp_anim = None
            return [i.img_path for i in saved_items]

    # Animate tab
    with gr.Tab("Anim") as longanimtab:
        gr.Markdown("""
        # Animation Station
        Samples points in latent space and lerps between them to create a looping animation.
        Uses the generation settings currently set in the main tab.
        """)

        longanim_root = "anims"
        longanims = []
        current_longanim = None

        with gr.Row():
            long_vid = gr.Video(interactive=False, autoplay=True, loop=True)

            # Settings for long vid - would be nice to have alternative setups too
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        la_num_images = gr.Number(4, step=1, label="Num samples", show_label=True)
                        frames_between_images = gr.Number(20, step=1, label="Frames between images", show_label=True)

                    with gr.Row():
                        la_batch_size = gr.Slider(1, 32, step=1, label="Batch size", show_label=True, value=4)
                        la_fps = gr.Number(20, step=1, label="FPS", show_label=True)
                        la_normalise = gr.Checkbox(value=True, label="Normalise", info="Whether to normalise intermediate latents to be the same size as the samples. Only uncheck if using StyleGAN with low W multiplier (e.g. 1)")

                    with gr.Row():
                        la_smooth = gr.Checkbox(value=True, label="Smooth path")
                        la_smooth_factor = gr.Number(2, step=0.1, label="Smooth factor", show_label=True, minimum=0)

                    make_long_anim = gr.Button("Animate", variant="primary")

                with gr.Group():
                    la_anim_name = gr.Textbox(value="Name", interactive=True)
                    la_save_anim = gr.Button("Save anim", variant='primary')

        # Loadable animations
        # textbox2 = gr.Textbox(interactive=False)
        la_anim_name_list = gr.Dataset(components=[la_anim_name], samples=[], type="values", label="Anims")

        @la_save_anim.click(inputs=[la_anim_name], outputs=[la_anim_name, la_anim_name_list])
        def change_anim_name(anim_name):
            global current_longanim, longanims

            if current_longanim is None:
                print("no anim to save")

            if os.path.exists(os.path.join(longanim_root, anim_name + ".mp4")):
                print("But that anim already exists")
                return

            shutil.move(join(longanim_root, current_longanim + ".mp4"), join(longanim_root, anim_name + ".mp4"))
            longanims.remove(current_longanim)
            longanims.append(anim_name)
            longanims = sorted(set(longanims))

            current_longanim = anim_name

            return current_longanim, gr.Dataset(samples=[[i] for i in longanims])

        @make_long_anim.click(inputs=gen_settings | {la_num_images, frames_between_images, la_batch_size, la_fps, la_normalise, la_smooth, la_smooth_factor}, outputs=[long_vid, la_anim_name, la_anim_name_list])
        def mk_longanim(data, progress=gr.Progress()):
            global current_longanim, longanims
            current_longanim = "tmp"

            num_images = data.pop(la_num_images)
            frames = data.pop(frames_between_images)
            batch_size = data.pop(la_batch_size)
            fps = data.pop(la_fps)
            normalise = data.pop(la_normalise)
            smooth = data.pop(la_smooth)
            smooth_factor = data.pop(la_smooth_factor)

            update_gan(progress=progress)
            config = config_from_settings(data)
            kwargs = gen_kwargs_from_settings(data)

            # normalise = "stylegan" not in gan.name

            zs = gan.get_z(num_images, **kwargs)
            if len(zs.shape) == 3:
                out = []
                k = zs.shape[1]
                for i in range(k):
                    p = interpolate(zs[:, i], gan.mean_latent, gan.std_latent, frames, normalise=normalise,
                                    gaussian_smooth=smooth, gaussian_sigma=smooth_factor)
                    out.append(p)
                zs = np.stack(out, axis=1)
            else:
                zs = interpolate(zs, gan.mean_latent, gan.std_latent, frames, normalise=normalise,
                                 gaussian_smooth=smooth, gaussian_sigma=smooth_factor)

            imgs = generate_from_config(config, zs, gan, upsampler, session, batch_size, progress)

            write_path = join(longanim_root, current_longanim + ".mp4")

            writer = util.AvVideoWriter(imgs[0].height, imgs[0].width, write_path, fps)
            for img in imgs:
                writer.write(img)
            writer.close()

            longanims.append(current_longanim)
            longanims = sorted(set(longanims))

            return write_path, current_longanim, gr.Dataset(samples=[[i] for i in longanims])

        @la_anim_name_list.select(outputs=[long_vid, la_anim_name])
        def select_anim_name(data: gr.SelectData):
            global current_longanim
            name = data.value[0]
            current_longanim = name
            return join(longanim_root, name + ".mp4"), name

        @longanimtab.select(outputs=la_anim_name_list)
        def refresh(data: gr.SelectData):
            global longanims
            longanims = sorted([i[:-4] for i in os.listdir(longanim_root) if i.endswith(".mp4")])
            return gr.Dataset(samples=[[i] for i in longanims])

demo.launch(share=False)
