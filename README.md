# super-mario-bros-reinforcement-learning
My implementation of an RL model to play the NES Super Mario Bros using Stable-Baselines3 (SB3). As of today (Aug 14 2022) the trained PPO agent completed World 1-1. 

The pre-trained models are located under ./models. To run these models run ./smb-ram-ppo-play.ipynb. 

To train a new model run ./smb-ram-ppo-train.ipynb.

![world-1-1-n_stack=4](https://user-images.githubusercontent.com/46117079/185268710-2d477eb0-b3f7-4ab8-865d-130e3a23a3e9.gif)

## Gym Environment

I used the [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) (4.3.0) environment and implemented a custom observation method that reads data from the game’s RAM map. The code can be found in ./smb_utils.py. In short:

*	The tiles (blocks, items, pipes, etc) are stored in 0x0500-0x0069F as a 32x13 grid (technically it’s two 16x13 grids but it’s not difficult to get the correct coordinates). The actual displayed grid (16x13) scrolls through this grid; once it reaches the end it wraps around and updates the memory grid (32x13) incrementally. Each tile corresponds to a 16x16 pixel square in the rendered screen, and since the top two rows (where the scores and other information are displayed) aren’t actually stored in the memory grid, we end up with 16x(13+2)*16 = 256x240 which is the pixel dimensions of the displayed screen. 

*	Mario & the enemies’ locations are represented by their x & y positions in unit of pixels rather than grid boxes. To fit them onto the grid I divide each pixel values by 16 and round to the nearest integers. 

*	Since there are a lot of tile and enemy types, to make things simpler I assigned an integer value to each group: 2 for Mario himself, 1 for all non-empty tiles, 0 for empty tiles, and -1 for enemies. This strategy worked out fine for world 1-1, but for later levels where there are non-stompable enemies like Piranha Plants or Spinies, the trained agent still treats them as Goombas and makes Mario committing suicides. 

The RAM access function is wrapped inside an ObservationWrapper. To add temporal information I also added a frame stack which returns the most recent n_stack number of frames, with each 2 separated by (n_skip – 1) frames. So without using cropping, the observation method returns a 3D array of shape (13, 16, n_stack). SB3 does have a VecFrameStack wrapper but I had trouble getting it working with my custom environment and so I wrote my own. 

As for why I used the RAM grid instead of the actual rendered screen as the input state, right now I can think of two reasons:

* First is obviously the computational cost. To process pixel images I would need to add several convolutional layers to extract the features first before pipe these features to a dense network. Training this convolutional network is going to be very computationally expensive, especially since all I have is a M1 MacBook Air. I did try to simplify the model through cropping and using only half of an image, but it still took me 8 hours to run for 100k steps and the agent didn’t appear to learn anything meaningful at all. Downsampling the images didn’t help either.
* Second is more of a design philosophy. I wanted to frame this project as “training an RL agent to understand the game’s mechanism in order to complete the level” which I considered to be separate from the object detection aspect of the overall problem. This would also mimic a more experienced human player who can tell an enemy apart from a brick tile or an item. To some degrees, the way I set up the custom environment does conflict with this idea (such as not distinguishing different enemy types which I actually mentioned earlier), but within the scope of this project I decided to keep things simple. 

I used the default reward function which is basically “how far did Mario travel in the level” minus the amount of clock time it took and an additional penalty if Mario died in the episode. Modifying the reward function would introduce additional degrees of freedom which there’s already plenty of in this project.

## Training & Results

I used the PPO agent in SB3 with the default paramenters and a linear learning rate scheduler to gradually reduce lr to 0 by the end of a session. Adding the scheduler significantly improved the stability of the training process. Each model was trained for 10M steps and took about 4.5 hours to finish.

<img src="https://user-images.githubusercontent.com/46117079/185277702-02190c31-7d88-4f71-a2f1-72f1bb879829.gif" width="400" >

Here’s the result of model pre-trained-1.zip (n_stack=4, n_skip=4, no cropping -- the same model shown on top). I wouldn’t say the model converged very well as the predicted actions still fluctuates a lot especially when Mario is airborne, but nonetheless Mario did complete the level at the end.

<img src="https://user-images.githubusercontent.com/46117079/185268750-4d273c40-9a4f-4367-96d4-dc0655ddbc7b.png" width="500" >



|                  |![world-1-1-n_stack=1](https://user-images.githubusercontent.com/46117079/185268999-1c00d0a6-643b-41d3-8e00-62dbcbc83746.gif)|![world-1-1-n_stack=2](https://user-images.githubusercontent.com/46117079/185269020-9f9c7abc-960e-42f4-95e7-78f88e23307e.gif)|![world-1-1-n_stack=4](https://user-images.githubusercontent.com/46117079/185269067-bdd0023d-e667-4baf-8365-df93dfa49b89.gif)|
|------------------|-------|------|------|
| n_stack          | 1     | 2    | 4    |
| Training steps   | 6.2M  | 9.6M | 10M  |
| Episode score    | 1723  | 3062 | 3066 |
| Episode steps    | 8019  | 1060 | 1118 |
| Completed level? | False | True | True |

So does adding temporal information actually matter? To test this I trained two more models using the same hyperparameters but changed n_stack to either 1 or 2. It turns out the 2-frame model actually completed the level and took even slightly less time/steps than the first model shown earlier. The single-frame model, on the other hand, gets stuck in front of a brick tile and stays there forever. To me it looks like having multiple frames doesn’t really help Mario at avoiding or stomping enemies – if Mario moves fast enough in the stage the enemies’ locations and paths are effectively identical in each run. However, it does help at preventing the agent from getting stuck in the level by letting it know that Mario has been stationary for too long and should try to jump over what’s blocking his path (see the 4-frame model at the final ramp). I wouldn’t be surprised if the single-frame model can actually complete the level given a little more training time, but using a frame stack does make the training process easier.

|![world-1-2-n_stack=4](https://user-images.githubusercontent.com/46117079/185269307-c35a2955-be07-4f4d-abad-9655829b03a5.gif)|![world-4-1-n_stack=4](https://user-images.githubusercontent.com/46117079/185269324-06919b57-2e03-46a1-b714-ce1133baf670.gif)|![world-5-1-n_stack=4](https://user-images.githubusercontent.com/46117079/185269337-b7744b45-679e-4c83-93b0-561af95d199d.gif)|
|---------------------------------------------------|--------------------------------------------|--------------------------------------------|
| 1-2: Mario can’t jump over a slightly higher wall | 4-1: Mario fails to jump over a wider pit. | 5-1: Mario tries to stomp a Piranha Plant. |

Unfortunately even though the agent completed level 1-1, it still struggled in other levels in various ways. To me it seems the agent didn’t learn every aspect of the game’s mechanism as I intended – it did figure out to hit the jump button before ramming into an enemy (see the 1-2 & 5-1 examples) or when seeing there’s there’s a bottomless pit in front (4-1 example although Mario didn’t make the jump). But then having completed 1-1 flawlessly vs failing every other stages suggests that the agent did rely on learning the level’s landscape in order to optimize a path from the starting position to the goal pole. I think these two aspects of learning are definitely related, but the prior one is going to be more useful when Mario encounters a different tile layout in a new level.

I can think of a few ways to address this problem, such as:

*	Random starting location in a level – I don’t think this is currently supported by this gym environment. We could try e.g. let the game run for some random number of steps first before letting the agent take over, but then the start position would still bias toward the earlier part of the stage. Still it could be something worth trying.

*	Transfer learning: use a pre-trained agent and train it on a new level for a less amount of steps/episodes. By the time this agent sees more levels it should supposedly act more like as if it knows how to play the game rather than memorizing a level’s layout. That however assumes it doesn’t forget what it learnt before as it trains on a different level.

*	Train on a subset of stages: in each episode we let the agent to train on a different stage selected randomly from a subset of (similar) levels. This is now supported after release 7.4.0 and would be very easy to implement. However I’m worried if the policy could actually converge but it worths trying nonetheless.

