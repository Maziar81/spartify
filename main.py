from kivy.uix.dropdown import DropDown
from kivy.uix.gridlayout import GridLayout
from kivymd.app import MDApp
from kivy.core.text import LabelBase
from kivy.utils import platform
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from keys import CID, CS
import spotipy
import torch
from spotipy.oauth2 import SpotifyOAuth
import requests
from diffusers import DiffusionPipeline
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivymd.uix.label import MDLabel
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.button import Button


# Material Design colors
PRIMARY_COLOR = (0.13, 0.59, 0.95, 1)
SECONDARY_COLOR = (0.96, 0.26, 0.21, 1)
BACKGROUND_COLOR = (0.88, 0.89, 0.87)

Window.clearcolor = (0.88, 0.89, 0.87)
Window.size = (524, 342)

class MaterialButton(Button):
    def __init__(self, **kwargs):
        super(MaterialButton, self).__init__(**kwargs)
        self.background_color = PRIMARY_COLOR
        self.color = (1, 1, 1, 1)
        self.font_size = 16
        self.size_hint_y = None
        self.height = 50
        self.border_radius = (25, 25, 25, 25)
        self.padding = (10, 10)


class MaterialTextInput(TextInput):
    def __init__(self, **kwargs):
        super(MaterialTextInput, self).__init__(**kwargs)
        self.background_color = (1, 1, 1, 1)
        self.padding = (10, 10)
        self.size_hint_y = None
        self.height = 50


class MaterialLabel(Label):
    def __init__(self, **kwargs):
        super(MaterialLabel, self).__init__(**kwargs)
        self.color = (0, 0, 0, 1)
        self.font_size = 18

class MaterialDropdownMenu:
    def __init__(self, caller, playlist_names):
        self.dropdown = DropDown()

        for name in playlist_names:
            btn = Button(text=name, size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
            self.dropdown.add_widget(btn)

        caller.bind(on_release=self.dropdown.open)
        self.dropdown.bind(on_select=lambda instance, x: setattr(caller, 'text', x))  # Change here

    def select_playlist(self, caller, selected_item):
        caller.text = selected_item



class MyApp(MDApp):
    def build(self):
        self.icon = "icon.png"
        self.title = "Spartify"
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=10)
        self.title_label = MaterialLabel(text="Welcome to Spartify", size_hint_y=None, height=50)
        self.layout.add_widget(self.title_label)

        self.description_label = MaterialLabel(text="Where you can create your playlist's artwork using generative AI",
                                               size_hint_y=None, height=50)
        self.layout.add_widget(self.description_label)

        self.button1 = MaterialButton(text="Create artwork for a playlist")
        self.button1.bind(on_press=self.create_artwork)
        self.layout.add_widget(self.button1)

        self.button2 = MaterialButton(text="View all playlists")
        self.button2.bind(on_press=self.view_playlists)
        self.layout.add_widget(self.button2)

        self.button3 = MaterialButton(text="Exit")
        self.button3.bind(on_press=self.exit_app)
        self.layout.add_widget(self.button3)

        return self.layout

    def on_start(self):
        # Set custom font to support emojis
        if platform != "android":
            font_path = "Apple Color Emoji.ttc"
            LabelBase.register(name="emojifont", fn_regular=font_path)
            self.theme_cls.font_styles["Body1"] = ["emojifont", 16, False, 0.15]
        else:
            # On Android, emoji font support is already included
            self.theme_cls.font_styles["Body1"] = ["Roboto", 16, False, 0.15]

    def create_artwork(self, instance):
        self.layout.clear_widgets()

        # Create a GridLayout to organize the widgets
        grid_layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        grid_layout.bind(minimum_height=grid_layout.setter('height'))

        # Add title and description labels to the GridLayout
        grid_layout.add_widget(self.title_label)
        grid_layout.add_widget(self.description_label)

        # Create a horizontal BoxLayout to hold the label and dropdown button
        playlist_layout = BoxLayout(size_hint_y=None, height=50)

        # Add playlist name label
        playlist_name_label = MaterialLabel(text="Select the playlist:", size_hint=(None, None), size=(200, 50))
        playlist_layout.add_widget(playlist_name_label)

        # Get playlist names
        playlist_names = get_all_names()  # Replace with your actual playlist names

        # Create caller button for dropdown menu
        self.caller_button = MaterialButton(text="Select Playlist", size_hint=(None, None), size=(200, 50))  # Store a reference
        dropdown_menu = MaterialDropdownMenu(self.caller_button, playlist_names)
        playlist_layout.add_widget(self.caller_button)

        # Add the playlist layout to the main layout
        grid_layout.add_widget(playlist_layout)

        # Create buttons for creating artwork and going back
        create_artwork_button = MaterialButton(text="Create artwork", size_hint=(None, None), size=(200, 50))
        create_artwork_button.bind(on_press=self.create_artwork_action)
        grid_layout.add_widget(create_artwork_button)

        back_button = MaterialButton(text="Back", size_hint=(None, None), size=(200, 50))
        back_button.bind(on_press=self.back_to_menu)
        grid_layout.add_widget(back_button)

        # Add the GridLayout to the main layout
        self.layout.add_widget(grid_layout)

    def create_artwork_action(self, instance):
        playlist_name = self.caller_button.text  # Access selected playlist name
        playlist_dict = get_playlist_dict(playlist_name)
        tt(playlist_name)
        print("Artwork created successfully!")

        # Open a popup window to preview the created image
        preview_image = Image(source="generated_image.png")
        popup = Popup(title='Artwork Preview', content=preview_image, size_hint=(None, None), size=(400, 400))
        popup.open()

    def view_playlists(self, instance):
        self.layout.clear_widgets()
        self.layout.add_widget(self.title_label)
        self.layout.add_widget(self.description_label)

        playlists = get_all_playlist_dict()
        for playlist in playlists:
            playlist_label = MDLabel(text=playlist['name'], font_style="Body1")
            self.layout.add_widget(playlist_label)

        back_button = MaterialButton(text="Back")
        back_button.bind(on_press=self.back_to_menu)
        self.layout.add_widget(back_button)

    def exit_app(self, instance):
        App.get_running_app().stop()

    def back_to_menu(self, instance):
        self.layout.clear_widgets()

        # Create new instances of labels
        self.title_label = MaterialLabel(text="Welcome to Spartify", size_hint_y=None, height=50)
        self.layout.add_widget(self.title_label)

        self.description_label = MaterialLabel(text="Where you can create your playlist's artwork using generative AI",
                                               size_hint_y=None, height=50)
        self.layout.add_widget(self.description_label)

        # Recreate buttons
        self.button1 = MaterialButton(text="Create artwork for a playlist")
        self.button1.bind(on_press=self.create_artwork)
        self.layout.add_widget(self.button1)

        self.button2 = MaterialButton(text="View all playlists")
        self.button2.bind(on_press=self.view_playlists)
        self.layout.add_widget(self.button2)

        self.button3 = MaterialButton(text="Exit")
        self.button3.bind(on_press=self.exit_app)
        self.layout.add_widget(self.button3)


def preprocess_prompt(prompt, tokenizer, max_length):
    tokenized_prompt = tokenizer(prompt, max_length=max_length, truncation=True, padding='max_length',
                                 return_tensors='pt')
    return tokenized_prompt


def tt(p: str):
    base = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to(device)
    refiner = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to(device)

    n_steps = 40
    high_noise_frac = 0.8

    prompt = p

    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    image.save("generated_image.png")


def get_image(name: str, url: str) -> None:
    response = requests.get(url)

    if response.status_code == 200:
        image_data = response.content

        filename = name + " image.jpg"

        with open(filename, "wb") as f:
            f.write(image_data)


def get_playlist_dict(name: str) -> dict:
    lp_dict = {'name': name}

    lp_search = spotipy.Spotify.search(sp, q=name, type='playlist')
    playlist_id = lp_search['playlists']['items'][0]['id']

    lp_list = spotipy.Spotify.playlist_items(sp, playlist_id)

    lp_dict['track_num'] = len(lp_list['items'])
    im = spotipy.Spotify.playlist_cover_image(sp, playlist_id)
    im = im[0]['url']
    lp_dict['image'] = im

    all_tracks = []

    for i in lp_list['items']:
        all_tracks.append((i['track']['artists'][0]['name'], i['track']['name']))

    lp_dict['tracks'] = all_tracks

    return lp_dict


def get_all_playlist_dict() -> list:
    all = []
    all_playlists = spotipy.Spotify.current_user_playlists(sp)

    for i in all_playlists['items']:
        info = spotipy.Spotify.playlist_items(sp, i['id'])
        all.append(get_playlist_dict(i['name']))

    return all

def get_all_names() -> list[str]:
    l = get_all_playlist_dict()
    names = []
    for i in l:
        names.append(i['name'])

    return names


if __name__ == '__main__':
    device = "mps"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CID,
                                                   client_secret=CS,
                                                   redirect_uri="http://localhost:8080/",
                                                   scope="user-library-read"))
    MyApp().run()
