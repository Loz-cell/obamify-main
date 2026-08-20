# Portraitify

Turn any uploaded image into the bundled portrait using the animated pixel-rearrangement engine from [Obamify](https://github.com/Spu7Nix/obamify).

The target portrait, app icons, built-in preset, browser metadata, and interface copy have been customized for this version. Processing runs locally in the browser; uploaded images are not sent to a server.

# How to use

1. Select **Transform a new image** and choose a JPG, PNG, or WebP file.
2. Use the zoom and position controls to frame the source image. The portrait on the right is the fixed target.
3. Select **Start**. After processing, replay the transformation or export it as a GIF.

Advanced settings:

| Setting               | Description                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------|
| resolution            | How many cells the images will be divided into. Higher resolution will capture more high frequency details. |
| proximity importance  | How much the algorithm changes the original image to make it look like the target image. Increase this if you want a more subtle transformation. |
| algorithm             | The algorithm used to calculate the assignment of each pixel. Optimal will find the mathematically optimal solution, but is extremely slow for high resolutions. |

# Building from source

1. Install [Rust](https://www.rust-lang.org/tools/install)
2. Run `cargo run --release` in the project folder

## Running the web version locally
1. Install [Rust](https://www.rust-lang.org/tools/install)
2. Install the required target with `rustup target add wasm32-unknown-unknown`
3. Install Trunk with `cargo install --locked trunk`
4. Run `trunk serve --release --open`

# Credits

The transformation engine and original application structure are from the MIT-licensed [Obamify project](https://github.com/Spu7Nix/obamify). See [LICENSE](LICENSE).
