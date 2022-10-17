# Chess-CV
**Computer vision project to recognize chess positions.**

<p align="center">
<img src="https://user-images.githubusercontent.com/56843532/135783923-31afa1f5-19a5-43dc-8687-28b3576dcf7f.png" width=75% height=75%>
</p>

## Description
This project uses computer vision tools and techniques to recognize a game state of a chess board based on its photo. Since the chess set we have is quite different from the ones in online datasets, we create our own dataset to train our model on. The assumption is that the picture is always taken from White's perspective, i.e. 1st rank should be in the bottom and the 8th rank should be at the top. The whole process can be divided into 4 main stages:
### 1. 2D-Projection
Given a picture of a chess board under an arbitrary angle, the goal is to convert it into a 2D-projection. This can be done using Hough line detection and Canny edge detection algorithms, with both available in `opencv`. The methods used in this part are very similar to those discussed in the [paper by Czyzewski, Laskowski, and Wasik (2020)](https://arxiv.org/pdf/1708.03898.pdf), and the related [GitHub project](https://github.com/maciejczyzewski/neural-chessboard). 
### 2. Board Splitting
Once a 2D-projection is obtained, the board is then split into 64 150x300 squares. Each square needs to be labeled on what piece (if any) is currently on it. For this step, we use multiple historical games and take a picture corresponding to every move in the game, then use data stored in .pgn files to find out what piece (if any) is currently on a given square. This means for every picture we take we get 64 labeled images, which allows us to collect over 50,000 images within a couple minutes (even though we spent a few hours taking ~800 photos, this is a great way to boost data collection since doing it manually would've required much more time). The similar method is used in [this article](https://tech.bakkenbaeck.com/post/chessvision#chessboard-recognition).
### 3. Convolutional Neural Network (CNN) Training
Once all squares are labeled, we split our labeled data into training (70%), validation (20%), and test (10%) data. We then construct a CNN with the following architecture (see `train.py` for reference):
* 5 convolutional (3x3) layers, each followed by a max-pooling layer (2x2)
* Fully-connected flat layer with 128 neurons and ReLU activation function
* Fully-connected output layer with 13 neurons (corresponding to 12 possible piece classes + empty) and softmax activation function
The model is trained for 10 epochs with a batch size of 16 samples. We then take the weights that produce the best performance on validation data. The resulting weights (included in `model_weights.h5`) produce accuracy of **88.9%** on test data.
### 4. Prediction
Finally, once we have our CNN trained, we write code to automate all previous steps (2D-projection -> Splitting into 64 squares -> Classifying each square -> Putting every predicted square together) to obtain an array representing a predicted board position, and do additional processing (Numpy array -> FEN string -> SVG file -> PNG file) to produce the final result, which we then save in the `results` folder.

## Requirements
The following modules must be installed to use this project:  
```bash
pip install keras  
pip install -U matplotlib  
pip install numpy  
pip install opencv-contrib-python  
pip install chess  
pip install scipy  
pip install tensorflow  
pip install wand
```

Note: Make sure to use the given command to install OpenCV (not just `pip install opencv-python`), otherwise some features might not work.

## Training
In order to train our model on a custom dataset, do the following:
1. Choose a historical game (e.g. Runau vs Schmidt 1972) and find a corresponding [PGN file](https://www.chessgames.com/perl/nph-chesspgn?text=1&gid=1507516). Copy the file into `data/raw/pgns/`.
2. Set your board into the initial position (as seen from White's perspective) and make the first move by White, then take a picture and label it as `1.jpg`. Put the picture into `data/raw/games/runau_schmidt/orig/`. Then, make the first move by Black, take a picture, label it as `2.jpg`, and put it in the same folder. Repeat the same for all remaining moves in the game.
 * **IMPORTANT:** Do NOT take a picture before the first move, i.e. with every piece in its initial position, as those aren't included in PGN files; make the first move THEN take the first picture
3. Flip the board (i.e. now you see it from Black's perspective) and repeat Step 2 except now you must save all pictures into `data/raw/games/runau_schmidt/rev/`. This is done to make sure our model isn't biased to perform better in recognizing pieces from White's perspective.
4. Repeat Steps 1-3 for as many games as you like, the more the better. In our model, we used 8 games (around 860 photos total).
5. Open `preprocess.py` and change elements of `game_list` to include the (folder) names of all games you've added in Steps 1-4 (e.g. `runau_schmidt`). 
6. Run `preprocess.py` to create 2D-projections of the games you took pictures of.
7. Run `create_labels.py` to split each 2D-projection into 64 pieces and label each of them according to the PGN file you provided in Step 1.
8. Run `split_data.py` to randomly split labels into train, validation, and test data (if needed, adjust the values of `TRAIN`, `VALIDATION`, `TEST` variables inside the file).
9. Run `train.py` to create and train a CNN. Again, you can tweak global variables to find what works best for you (e.g. the number of epochs, batch size, etc). 
10. Run `main.py img1 img2 ...` where `img1 img2 ...` are paths to the images you would want to analyze, separated by space (the model weights are loaded from what was saved at Step 9). Check the `results` folder for the final output - the names will correspond to those of the original images so make sure every image you provide has a unique name. Enjoy!

## Labeling
If you want to use the provided model to analyze a custom set of images, simply do Step 10 of the **Training** part (no need to do previous steps).

## Additional Files
The project comes with 3 examples marked `Ex1.jpg`, `Ex2.jpg`, and `Ex3.jpg`, chosen so that they cover all pieces. The full set of ~800 photos of 8 games used for training and testing the model can be found [here](https://drive.google.com/file/d/1Ynge-pn1szzJlmQPIAd1BMGaSkOw_vly/view?usp=sharing). Unzip the archive and move the `games` folder into `data/raw/` (don't forget to obtain PGN files for these 8 games and put them in `data/raw/pgns/`). 

## Features
* Unlike many other projects, this project does not try to contextualize the position (e.g. by using a chess engine to calculate probability of a given position or by keeping history of previous positions), which means there is no guessing involved, only classification; it can also identify positions that are technically illegal/impossible to get in a real game (like `Ex3.jpg`)
* This project does not assume Pawns always promoting to Queens, which might be crucial in situations like the Saavedra position (where promoting to Queen leads to draw whereas underpromoting to Rook leads to victory) or the game between Runau and Schmidt in 1972 (where underpromoting to Knight allowed Runau to checkmate his opponent)


## Disadvantages
* Because of the way 2D-projection is done, the top half of pieces that are on the 8th rank is often cropped out making it much harder to classify them correctly, so this tool works better with pieces in the center or on the other side of the board
* Sometimes the tool doesn't correctly identify a piece that is obstructed by another piece, or it mistaks the top half of another square for a separate piece and gives a false positive
* In the chess set we used, King and Queen are hard to distinguish even by eye, and our tool seems to often mix up these two; because of this, we implemented the algorithm where, if we did not find a King of a certain color on the board, we look for a same-colored Queen and replace it with King in our prediction; this way of dealing with a problem has disadvantages (e.g. what if there are 2 or more Queens classified? what if no Queens were found but the missing King was classified as a Rook?) but it works in some situations, especially when there aren't many pieces left on the board
* This tool works better when there are fewer pieces on the board

## Possible Improvements
* We could possibly try out different architectures to achieve better accuracy (we tried VGG19 and ResNet50 but they didn't work well with our data, at least with computational resources we had), especially between King and Queen
* Because the method we use crops out the top half of pieces on the 8th rank, we could modify the cropping method to preserve the information about these pieces to make their classification more accurate
* We could potentially preprocess each square so that we first ignore the color of the piece and classify the piece and only then find its color (where we could use the fact that we always know the color of each square)

## References
* The 2D-projection part is heavily based on https://arxiv.org/pdf/1708.03898.pdf and the related project: https://github.com/maciejczyzewski/neural-chessboard
* Collecting data: https://tech.bakkenbaeck.com/post/chessvision#chessboard-recognition
* CNN architecture: https://towardsdatascience.com/a-single-function-to-streamline-image-classification-with-keras-bd04f5cfe6df
* https://par.nsf.gov/servlets/purl/10099572
* https://towardsdatascience.com/board-game-image-recognition-using-neural-networks-116fc876dafa
* https://web.stanford.edu/class/cs231a/prev_projects_2016/CS_231A_Final_Report.pdf
* https://cvgl.stanford.edu/teaching/cs231a_winter1415/prev/projects/chess.pdf
* https://github.com/bakkenbaeck/chessboardeditor
* https://github.com/jialinding/ChessVision
* https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
* https://stackoverflow.com/questions/56754543/generate-chess-board-diagram-from-an-array-of-positions-in-python
* https://www.programcreek.com/python/?code=yaqwsx%2FPcbDraw%2FPcbDraw-master%2Fpcbdraw%2Fpcbdraw.py#
* https://chess.stackexchange.com/questions/28870/render-a-chessboard-from-a-pgn-file
