import numpy as np
import random

class Osero:
    Width = 8
    Height = 8

    #white:-1, black:1, empty:0
    Field = None
    Depth = 4
    SelfColor = None
    StoneMap = None

    def __init__(self):
        self.Field = [[-1 for _ in range(self.Width)] for _ in range(self.Height)]
        self.Field[3][3] = 0
        self.Field[3][4] = 1
        self.Field[4][3] = 1
        self.Field[4][4] = 0

    def Disp(self,field):
        s = "  "
        for i in range(self.Width):
            s += "|"+str(i)+" "
        s += "\n  "
        for i in range(self.Width):
            s += "|--"
        s += "\n"
        for i in range(self.Height):
            s += str(i) + " "
            for j in range(self.Width):
                if field[i][j] == 0:
                    s += "|● "
                if field[i][j] == 1:
                    s += "|○ "
                if field[i][j] == -1:
                    s += "|  "
            s += "|\n"
        s+= "  "
        for i in range(self.Width):
            s += "|--"
        s += "\n"
        print(s)

    def Put(self,y,x,color,field):
        dxArr = [0,1,1,1,0,-1,-1,-1]
        dyArr = [1,1,0,-1,-1,-1,0,1]
        for dx,dy in zip(dxArr,dyArr):
            if(l := self.CheckDepth(color,field,y,x,dy,dx),l!=-1):
                xx = x
                yy = y
                for i in range(l+1):
                    field[yy][xx] = color
                    xx += dx
                    yy += dy
        return field
    
    def CanPUt(self,y,x,color,field):
        if not self.IsInside(y, x) or field[y][x] != -1:
            return False
        dxArr = [0,1,1,1,0,-1,-1,-1]
        dyArr = [1,1,0,-1,-1,-1,0,1]
        for dx,dy in zip(dxArr,dyArr):
            if(self.CheckDepth(color,field,y,x,dy,dx) > 0):
                return True
    
    def CheckDepth(self,color,field,y,x,dy,dx):
        depth = 0
        while(True):
            y += dy
            x += dx
            if(self.IsInside(y,x)==False or field[y][x] == -1):
                return -1
            if(field[y][x] == color):
                return depth
            depth += 1

    def GetPossiblePutPosition(self,color,field):
        positions = []
        for y in range(self.Height):
            for x in range(self.Width):
                if self.CanPUt(y,x,color,field):
                    positions.append([y,x])
        return positions
        
    def IsInside(self,y,x):
        return x >= 0 and x < self.Width and y >= 0 and y < self.Height

    def IsGameOver(self,color,field):
        return  not any(-1 in _ for _ in field) or (not self.GetPossiblePutPosition(color,field) and not self.GetPossiblePutPosition((color + 1)%2,field))
    
    def Result(self):
        n_white = 0
        n_black = 0
        n_sum = 0
        for y in range(self.Height):
            for x in range(self.Width):
                if self.Field[y][x] == -1:
                    continue
                n_sum += 1
                if self.Field[y][x] == 0:
                    n_white += 1
                if self.Field[y][x] == 1:
                    n_black += 1
        return n_white,n_black,n_sum
    
    def CalcStoneValue(self,color,field,stoneMap):
        value = 0
        for y in range(self.Height):
            for x in range(self.Width):
                if field[y][x] == color:
                    value += stoneMap[y][x]
                if field[y][x] == (color + 1)%2:
                    value -= stoneMap[y][x]
        return value

    def Alphabeta(self,color,field,depth,alpha,beta):
        if depth == 0 or self.IsGameOver(color,field):
            return self.CalcStoneValue(self.SelfColor,field,self.StoneMap)   
        
        positions = self.GetPossiblePutPosition(color,field)
        if len(positions) == 0:
            return self.Alphabeta((color + 1)%2,field,depth - 1,alpha,beta)

        if color == self.SelfColor:
            maxValue = float('-inf')
            for arr in positions:
                newField = self.Put(arr[0],arr[1],color,np.copy(field))
                value = self.Alphabeta((color + 1)%2,newField,depth - 1,alpha,beta)
                maxValue = max(maxValue,value)
                alpha = max(alpha,value)
                if beta <= alpha:
                    break
            return maxValue
        else:
            minValue = float('inf')
            for arr in positions:
                newField = self.Put(arr[0],arr[1],color,np.copy(field))
                value = self.Alphabeta((color + 1)%2,newField,depth - 1,alpha,beta)
                minValue = min(minValue,value)
                beta = min(beta,value)
                if beta <= alpha:
                    break
            return minValue

    def AI_move(self,color,stoneMap):
        self.SelfColor = color
        self.StoneMap = stoneMap
        positions = self.GetPossiblePutPosition(color,self.Field)
        best_hand = [positions[0]]
        maxValue = float('-inf')

        for arr in positions:
            newField = self.Put(arr[0],arr[1],color,np.copy(self.Field))
            value = self.Alphabeta((color + 1)%2,newField,self.Depth - 1,float('-inf'),float('inf'))
            if value > maxValue:
                maxValue = value
                best_hand = [arr]
            elif value == maxValue:
                best_hand.append(arr)
        position = random.choice(best_hand)
        self.Put(position[0],position[1],color,self.Field)

    def Player_move(self,color):
        positions = self.GetPossiblePutPosition(color,self.Field)
        print("おける場所")
        for (yy,xx) in positions:
            print("yy:",yy,"xx:",xx)
        while True:
            y = int(input("y:"))
            x = int(input("x:"))
            if not self.CanPUt(y, x, color,self.Field):
                print("置やんぞ!!")
            else:
                self.Put(y,x,color,self.Field)
                break

def PlayOsero(stoneMap1,stoneMap2):
    o = Osero()
    color = 0
    while not o.IsGameOver(np.copy(color),o.Field):
        # o.Disp(o.Field)
        positions = o.GetPossiblePutPosition(color,o.Field)
        if len(positions) == 0:
            color = (color + 1)%2
            continue
        if color == 0:
            # print("●の番です")
            # o.Player_move(color)
            o.AI_move(color,stoneMap1)
        else:
            # print("○の番です")
            o.AI_move(color,stoneMap2)
        color = (color + 1)%2
    # o.Disp(o.Field)
    n_white,n_black,n_sum = o.Result()
    # print("white:",n_white,"black:",n_black,"total:",n_sum)
    return o.Result()
            

if __name__ == "__main__":
    stoneMap = [
        [30, -12, 0, -1, -1, 0, -12, 30],
        [-12, -15, -3, -3, -3, -3, -15, -12],
        [0, -3, 0, -1, -1, 0, -3, 0],
        [-1, -3, -1, -1, -1, -1, -3, -1],
        [-1, -3, -1, -1, -1, -1, -3, -1],
        [0, -3, 0, -1, -1, 0, -3, 0],
        [-12, -15, -3, -3, -3, -3, -15, -12],
        [30, -12, 0, -1, -1, 0, -12, 30]
    ]

    PlayOsero(stoneMap,stoneMap)