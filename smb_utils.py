#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 18:52:21 2022

@author: yumouwei
"""

import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class smb_grid:
    
    def __init__(self, env):
        self.ram = env.unwrapped.ram
        self.screen_size_x = 16     # rendered screen size
        self.screen_size_y = 13
        
        self.mario_level_x = self.ram[0x6d]*256 + self.ram[0x86]
        self.mario_x = self.ram[0x3ad]  # mario's position on the rendered screen
        self.mario_y = self.ram[0x3b8] + 16 # top edge of (big) mario
        
        self.x_start = self.mario_level_x - self.mario_x # left edge pixel of the rendered screen in level
        self.rendered_screen = self.get_rendered_screen()
        
    
    ########
    # get background tile grid
    
    def tile_loc_to_ram_address(self, x, y):
        '''
        convert (x, y) in Current tile (32x13, stored as 16x26 in ram) to ram address
        x: 0 to 31
        y: 0 to 12
        '''
        page = x // 16
        x_loc = x%16
        y_loc = page*13 + y
        
        address = 0x500 + x_loc + y_loc*16
        
        return address

    def get_rendered_screen(self):
        '''
        Get the rendered screen (16 x 13) from ram
        empty: 0
        tile: 1
        enemy: -1
        mario: 2
        '''
        
        # Get background tiles
        
        rendered_screen = np.zeros((self.screen_size_y, self.screen_size_x))
        screen_start = int(np.rint(self.x_start / 16))
    
        for i in range(self.screen_size_x):
            for j in range(self.screen_size_y):
                x_loc = (screen_start + i) % (self.screen_size_x * 2)
                y_loc = j
                address = self.tile_loc_to_ram_address(x_loc, y_loc) 
                #bg_screen2[j, i] = env.unwrapped.ram[address]
                
                # Convert all types of tile to 1
                if self.ram[address] != 0:
                    rendered_screen[j, i] = 1
                    
        # Add mario
        x_loc = (self.mario_x + 8) // 16
        y_loc = (self.mario_y - 32) // 16 # top 2 rows in the rendered screen aren't stored in ram
        if x_loc < 16 and y_loc < 13:
            rendered_screen[y_loc, x_loc] = 2
        
        # Add enemies
        for i in range(5):
            # check if the enemy is drawn
            if self.ram[0xF + i] == 1: 
                enemy_x = self.ram[0x6e + i]*256 + self.ram[0x87 + i] - self.x_start
                enemy_y = self.ram[0xcf + i]
                x_loc = (enemy_x + 8) // 16
                y_loc = (enemy_y + 8 - 32) // 16

                # check if the enemy is inside the rendered screen
                # 8/6/22 fixed bug where enemy with x_loc < 0 still got added to rendered_screen; doesn't seem to affect trained models' performance
                # if x_loc < 16 and y_loc < 13: 
                if 0 <= x_loc < 16 and 0 <= y_loc < 13: 
                    rendered_screen[y_loc, x_loc] = -1
                
        return rendered_screen
