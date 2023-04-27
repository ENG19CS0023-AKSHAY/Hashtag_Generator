from tkinter import *
from tkinter import filedialog
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import ImageTk, Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def predict_step(image_paths, num_captions=15):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences": num_captions}
    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    preds = [pred.strip() for pred in preds]
    return preds

def CountFreq(li):
    freq = {}
    for items in li:
        a = items.split()
        for i in a:
            if len(i) > 3:
                freq[i] = li.count(i)
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    top_10_hashtags = [f"#{hashtag}" for hashtag, count in sorted_freq[:10]]
    return top_10_hashtags

def select_image():
    global panel
    global image_path
    image_path = filedialog.askopenfilename()
    if len(image_path) > 0:
        image = Image.open(image_path)
        image = image.resize((250, 250), resample=Image.LANCZOS)
        image = ImageTk.PhotoImage(image)
        panel.config(image=image)
        panel.image = image

def generate_hashtags():
    global image_path
    if image_path is None:
        return
    try:
        li = predict_step([image_path], num_captions=8)
        top_10_hashtags = CountFreq(li)
        hashtags_text.set('\n'.join(top_10_hashtags))
    except:
        hashtags_text.set('Error generating hashtags')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 16
num_beams = 20

root = Tk()
root.title("Image Hashtag Generator")

frame = Frame(root)
frame.pack()

panel = Label(frame)
panel.pack(side="left", padx=10, pady=10)

select_button = Button(frame, text="Select Image", command=select_image)
select_button.pack(side="bottom", pady=10)

generate_button = Button(frame, text="Generate Hashtags", command=generate_hashtags)
generate_button.pack(side="bottom", pady=10)

hashtags_text = StringVar()
'''hashtags_text.set('No hashtags generated yet')'''

hashtags_label = Label(root, textvariable=hashtags_text, font=("Arial", 14), wraplength=400, justify="left")
hashtags_label.pack(padx=10, pady=10)

root.mainloop()
