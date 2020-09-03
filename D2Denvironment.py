import numpy as np
import random
import math


np.random.seed(1234)


class D2Dchannels:
    
    def __init__(self):
        
        self.bandwidth = 1E6
        self.alpha = 2
        self.noise_dB = -114;
        self.noise_pow = 10**(self.noise_dB/10)
        self.fsPL = 10**(-61.7/10)
        self.mean = 7
        self.std = 1/7
        
    def get_channel(self,numUsers):  ### channel fading model
    
        channel = np.zeros((numUsers, numUsers))
        for i in range(numUsers):
            for j in range(i, numUsers):
                channel[i,j] = channel[j,i] = np.random.gamma(self.mean, self.std)

        return channel
    
    
class Antenna:
    
    def __init__(self):
        #, main_lobe, side_lobe, sectbW, bWlb, bWub, bWNumb, Tpilot
        
        self.bWlb = math.radians(10)  
        self.bWub = math.radians(60)
        self.bWNumb = 10
        self.Tpilot = 1.5E-2
        self.sectbW = math.radians(60)
        self.aplpha = 0.9  ### threshold for link stability
    
        
        self.beamwidth = np.linspace(self.bWlb, self.bWub, self.bWNumb)    ### theta_3dB
        self.align_time = self.Tpilot*(self.sectbW/self.beamwidth)**2
        self.main_lobe = (math.pi*10**(2.028))/(42.6443*self.beamwidth/2+math.pi)
        self.side_lobe = 10**(-2.028)* self.main_lobe  
        
        
    def AntGain (self, TxLoc, RxLoc, TxhpBW, RxhpBW, TxTheta, RxTheta):

        phi = math.atan(abs(TxLoc[1]-RxLoc[1] /max(.001,abs(TxLoc[0]-RxLoc[0]))))
        beamind_tx = np.where(self.beamwidth == TxhpBW)
        beamind_rx = np.where(self.beamwidth == RxhpBW)
        ### First quarter
        if TxLoc[0] > RxLoc[0] and TxLoc[1] >= RxLoc[1] :
            
            if abs(phi-RxTheta) < RxhpBW/2 :
                GRx = self.main_lobe[beamind_rx]
            else:
                GRx = self.side_lobe[beamind_rx]
                
            if abs(math.pi+phi-TxTheta)<TxhpBW/2 :
                GTx = self.main_lobe[beamind_tx]
            else:
                GTx = self.side_lobe[beamind_tx]
            
        
        ### 2nd quarter       
        elif TxLoc[0] <= RxLoc[0] and TxLoc[1]> RxLoc[1]:
            
            if abs(math.pi-phi-RxTheta) < RxhpBW/2 :
                GRx = self.main_lobe[beamind_rx]
            else:
                GRx = self.side_lobe[beamind_rx]
                
            if abs(2*math.pi-phi-TxTheta) < TxhpBW/2 :
                GTx = self.main_lobe[beamind_tx]
            else:
                GTx = self.side_lobe[beamind_tx]
                 
        
        ### 3rd quarter       
        elif TxLoc[0] < RxLoc[0] and TxLoc[1] <= RxLoc[1]:
            
            if abs(math.pi+phi-RxTheta) < RxhpBW/2 :
                GRx = self.main_lobe[beamind_rx]
            else:
                GRx = self.side_lobe[beamind_rx]
                
            if abs(phi-TxTheta) < TxhpBW/2 :
                GTx = self.main_lobe[beamind_tx]
            else:
                GTx = self.side_lobe[beamind_tx]
            
                   
        ### 4th quarter       
        else:# TxLoc[0] >= RxLoc[0] and TxLoc[1] < RxLoc[1]:
            
            if abs(2*math.pi-phi-RxTheta) < RxhpBW/2 :
                GRx = self.main_lobe[beamind_rx]
            else:
                GRx = self.side_lobe[beamind_rx]
                
            if abs(phi-TxTheta) < TxhpBW/2 :
                GTx = self.main_lobe[beamind_tx]
            else:
                GTx = self.side_lobe[beamind_tx]
            
        return GRx, GTx
    
    
class D2D_links:
    
    def __init__(self, txLoc, rxLoc, txVelocity, rxVelocity, mDirtx, mDirrx, txAntAng, rxAntAng, link_dis, bW):
        self.txLoc = txLoc
        self.rxLoc = rxLoc
        self.txVelocity = txVelocity
        self.rxVelocity = rxVelocity
        self.mDirtx = mDirtx
        self.mDirrx = mDirrx
        self.txAntAng = txAntAng
        self.rxAntAng = rxAntAng
        self.link_dis = link_dis
        self.bW = bW
        
        

class Environmnet:
    
    def __init__(self, numUsers, width, length, vlb, vup, distancelb, distanceup, delta_theta):
        self.numUsers = numUsers
        self.width = width
        self.length = length
        self.vlb = vlb
        self.vup = vup
        self.distancelb = distancelb 
        self.distanceup = distanceup
        self.delta_theta = math.radians(delta_theta)
        
        self.Antenna = Antenna()
        self.channel = D2Dchannels()
        
        self.D2Dlinks = []
        self.D2Ddata = []
        
        self.power = 1
        self.delta_tau = .1  ## Agen can select antenna every delta_tau
        self.horizon = 10 ## episod horizon is 100 ms
        # self.data_size = int((4 * 190 + 300) * 8 * 2)
        self.data_size = 35E6
        self.ch_mtx = self.channel.get_channel(self.numUsers) 
        self.link_time = np.zeros((self.numUsers,1))
        self.bAtime = np.zeros((self.numUsers,1))
        self.action_list = self.Antenna.beamwidth

    def add_new_link(self):
        
        self.velocity_tx = np.random.uniform(self.vlb, self.vup, (self.numUsers,1))
        self.velocity_rx = np.random.uniform(self.vlb, self.vup, (self.numUsers,1))
        self.angle = np.random.uniform(0, math.radians(360), (self.numUsers,1))  ## Direction of antenna
        self.linkDis = np.random.uniform (self.distancelb, self.distanceup,(self.numUsers,1))
        self.angle_mtx = np.random.uniform(0, math.radians(360), (self.numUsers,1))  ## Direction of movement of tx
        self.angle_mrx = np.random.uniform(0, math.radians(360), (self.numUsers,1))  ## Direction of movement or rx

        
        txLocx = np.random.uniform(-self.width/2, self.width/2, (self.numUsers,1))
        txLocy = np.random.uniform(-self.length/2, self.length/2, (self.numUsers,1))
        self.txLoc = np.concatenate((txLocx, txLocy),axis=1)
        
        self.rxLoc = np.zeros((self.numUsers,2))
        self.beam = np.zeros((self.numUsers,1))
        self.rxAng = np.zeros((self.numUsers,1))
        self.beamInd = np.zeros((self.numUsers,1),dtype=np.int8)
        self.intf_ind = np.zeros((self.numUsers,self.numUsers))
        self.intf = np.zeros((self.numUsers,1))
        self.penalty = np.zeros((self.numUsers,1))
        
        for i in range (self.numUsers):
            self.beamInd[i] = random.randint(0, len(self.Antenna.beamwidth)-1)
            self.beam[i,0] = self.Antenna.beamwidth[self.beamInd[i,0]]
            trans = np.array([[math.cos(self.angle[i,0])*self.linkDis[i,0],math.sin(self.angle[i,0])*self.linkDis[i,0]]])
            self.rxLoc[i,:] = np.ravel( np.array([[self.txLoc[i,0]+trans[0,0],self.txLoc[i,1]+trans[0,1]]]))
            self.rxAng[i] = math.atan2(-self.rxLoc[i,1] + self.txLoc[i,1], -self.rxLoc[i,0] + self.txLoc[i,0])
            # if self.txLoc[i,1] <= self.rxLoc[i,1]:
            #     self.rxAng[i] = self.angle[i,0] + math.pi
            # else:
            #     self.rxAng[i] = self.angle[i,0] - math.pi
            
        for i in range(self.numUsers):
            txLoc = self.txLoc[i,:]
            rxLoc = self.rxLoc[i,:]
            txVelocity = self.velocity_tx[i,0]
            rxVelocity = self.velocity_rx[i,0]
            mDirtx = self.angle_mtx[i,0]
            mDirrx = self.angle_mrx[i,0]
            txAntAng = self.angle[i,0]
            rxAntAng = self.rxAng[i,0]
            link_dis = self.linkDis[i,0]
            bW = self.beam[i,0]
            self.D2Dlinks.append(D2D_links(txLoc, rxLoc, txVelocity, rxVelocity, mDirtx, mDirrx, txAntAng, rxAntAng, link_dis, bW))
            
    def renew_location(self, curent_location, velocity, mov_angle, delta_t, delta_theta): ### random_walk renew the location of each user
        
        theta = mov_angle +(-delta_theta + 2*delta_theta*np.random.uniform(0, 1))
        x = curent_location[0] + velocity*delta_t*math.cos(theta)
        y = curent_location[1] + velocity*delta_t*math.sin(theta)
        
        if x >= self.length :
            x = min(self.length, x)
        elif x <= -self.length:
            x = max(-self.length, x)
            
            
        if y >= self.width:
            y = min(self.width, y)
        elif y <= -self.width:
            y = max(-self.width, y)
            
        return (np.array([x,y]), theta)
    
    def renew_trajectory(self):
        
        for i in range(self.numUsers):
            self.D2Dlinks[i].txLoc,  self.D2Dlinks[i].mDirtx = self.renew_location(self.D2Dlinks[i].txLoc, self.D2Dlinks[i].txVelocity, self.D2Dlinks[i].mDirtx, self.delta_tau, self.delta_theta )
            self.D2Dlinks[i].rxLoc, self.D2Dlinks[i].mDirrx = self.renew_location(self.D2Dlinks[i].rxLoc, self.D2Dlinks[i].rxVelocity, self.D2Dlinks[i].mDirrx, self.delta_tau, self.delta_theta)
            ### Update link length:
            self.D2Dlinks[i].link_dis = max(0.01, math.hypot(self.D2Dlinks[i].rxLoc[0]-self.D2Dlinks[i].txLoc[0], self.D2Dlinks[i].rxLoc[1]-self.D2Dlinks[i].txLoc[1]))
            self.linkDis[i] = self.D2Dlinks[i].link_dis 
            #### Update antenna angle for beam alignment:
            self.D2Dlinks[i].txAntAng = math.atan2(self.D2Dlinks[i].rxLoc[1]-self.D2Dlinks[i].txLoc[1], self.D2Dlinks[i].rxLoc[0]-self.D2Dlinks[i].txLoc[0])
            self.D2Dlinks[i].rxAntAng = math.atan2(-self.D2Dlinks[i].rxLoc[1]+self.D2Dlinks[i].txLoc[1], -self.D2Dlinks[i].rxLoc[0]+self.D2Dlinks[i].txLoc[0])

            
    def renew_channel(self):
        
        self.ch_mtx = self.channel.get_channel(self.numUsers) 
        
   
    def compute_link_timing(self, action): ### rx is considered as the reference
        
        for l in range(self.numUsers):
            v_rel_x = self.D2Dlinks[l].rxVelocity * math.sin(self.D2Dlinks[l].mDirrx)-self.D2Dlinks[l].txVelocity * math.sin(self.D2Dlinks[l].mDirtx)
            v_rel_y = self.D2Dlinks[l].rxVelocity * math.cos(self.D2Dlinks[l].mDirrx)-self.D2Dlinks[l].txVelocity * math.cos(self.D2Dlinks[l].mDirtx)
            
            v_rel = math.hypot(v_rel_x, v_rel_y)
            ang_rel = math.atan2(v_rel_y, v_rel_x)
            
            self.link_time[l] = abs((self.D2Dlinks[l].link_dis*self.Antenna.beamwidth[action[l]])/(v_rel*math.sin(ang_rel)))*math.sqrt(math.log(1/self.Antenna.aplpha)/(0.3*math.log(10)))
            
            self.bAtime[l] = self.Antenna.align_time[action[l]]
        
    
    def compute_reward(self, action):
        self.compute_link_timing(action)
        signal = np.zeros((self.numUsers,1))
        sinr = np.zeros((self.numUsers,1))
        rewards = np.zeros((self.numUsers,1))
        self.penalty = np.zeros((self.numUsers,1))
        self.distan_mat = np.zeros((self.numUsers, self.numUsers))
        rate = np.zeros((self.numUsers,1))
        self.intf = np.zeros((self.numUsers,1))
        for rx in range (self.numUsers):
            for tx in range (self.numUsers):
                if tx != rx:
                    self.distan_mat[rx,tx]  = max(0.01, math.hypot(self.D2Dlinks[rx].rxLoc[0]-self.D2Dlinks[tx].txLoc[0] , self.D2Dlinks[rx].rxLoc[1]-self.D2Dlinks[tx].txLoc[1]))
                
        for rx in range (self.numUsers):
            intereference = np.zeros((self.numUsers,1))
            beamind = action[rx]
            signal[rx] = self.power*self.ch_mtx[rx,rx]*self.Antenna.main_lobe[beamind]**2*self.channel.fsPL*self.D2Dlinks[rx].link_dis**(-self.channel.alpha)
            
            self.penalty[rx] = max( 1 -(self.bAtime[rx]/ min(self.delta_tau, self.link_time[rx])), 0)
            for tx in range (self.numUsers):
                if tx == rx:
                    intereference[tx] = 0
                elif tx != rx and self.active_links[tx]:
                    Gtx,Grx = self.Antenna.AntGain (self.D2Dlinks[tx].txLoc, self.D2Dlinks[rx].rxLoc, self.Antenna.beamwidth[action[tx]], self.Antenna.beamwidth[action[rx]], self.D2Dlinks[tx].txAntAng, self.D2Dlinks[rx].rxAntAng)
                    intereference [tx] = self.power*self.ch_mtx[rx,tx]*Gtx*Grx*self.channel.fsPL*self.distan_mat[rx,tx]**(-self.channel.alpha)
                    self.intf_ind[rx,tx] = 10*np.log10(intereference [tx]+ (self.channel.noise_pow/4))
            sumInt = np.sum(intereference) + self.channel.noise_pow
            self.intf[rx] = 10*np.log10(sumInt)
            sinr[rx] = signal[rx]/ sumInt
            # rewards[rx] = self.channel.bandwidth * np.log2(1 + sinr[rx]) * self.penalty[rx]/10E7
            rate[rx] = self.channel.bandwidth * np.log2(1 + sinr[rx]) * self.penalty[rx]
            self.data[rx] -= rate[rx] * min(self.delta_tau, self.link_time[rx])
            rewards[rx] = min(1,rate[rx] * min(self.delta_tau, self.link_time[rx])/self.data_size) ### Amunt of data sent in each time step is the individual reward of the agent)
            self.individual_time_limit[rx] -= self.delta_tau

            
        self.data[self.data < 0] = 0
        # rewards[self.data == 0] = 1
        self.active_links[np.multiply(self.active_links, self.data <= 0)] = 0
        return rate, rewards
    
    def compute_reward_test(self, action):
        self.compute_link_timing(action)
        signal = np.zeros((self.numUsers,1))
        sinr = np.zeros((self.numUsers,1))
        rate = np.zeros((self.numUsers,1))
        self.penalty = np.zeros((self.numUsers,1))
        # self.distan_mat = np.zeros((self.numUsers, self.numUsers))
        
        # for rx in range (self.numUsers):
        #     for tx in range (self.numUsers):
        #         if tx != rx:
        #             self.distan_mat[rx,tx]  = min(0.01, math.hypot(self.D2Dlinks[rx].rxLoc[0]-self.D2Dlinks[tx].txLoc[0] , self.D2Dlinks[rx].rxLoc[1]-self.D2Dlinks[tx].txLoc[1]))
                
        for rx in range (self.numUsers):
            intereference = np.zeros((self.numUsers,1))
            beamind = action[rx]
            signal[rx] = self.power*self.ch_mtx[rx,rx]*self.Antenna.main_lobe[beamind]**2*self.channel.fsPL*self.D2Dlinks[rx].link_dis**(-self.channel.alpha)
            
            self.penalty[rx] = max( 1 -(self.bAtime[rx]/ min(self.delta_tau, self.link_time[rx])), 0)
            for tx in range (self.numUsers):
                if tx == rx:
                    intereference[tx] = 0
                elif tx != rx and self.active_links_rand[tx]:
                    Gtx,Grx = self.Antenna.AntGain (self.D2Dlinks[tx].txLoc, self.D2Dlinks[rx].rxLoc, self.Antenna.beamwidth[action[tx]], self.Antenna.beamwidth[action[rx]], self.D2Dlinks[tx].txAntAng, self.D2Dlinks[rx].rxAntAng)
                    intereference [tx] = self.power*self.ch_mtx[rx,tx]*Gtx*Grx*self.channel.fsPL*self.distan_mat[rx,tx]**(-self.channel.alpha)
            sumInt = np.sum(intereference) + self.channel.noise_pow
            sinr[rx] = signal[rx]/ sumInt
            rate[rx] = self.channel.bandwidth * np.log2(1 + sinr[rx]) * self.penalty[rx]
            self.data_rand[rx] -= rate[rx] * min(self.delta_tau, self.link_time[rx])
            self.individual_time_limit[rx] -= self.delta_tau
        self.data_rand[self.data_rand < 0] = 0
        # rate[self.data_rand == 0] = 1
        self.active_links_rand[np.multiply(self.active_links_rand, self.data_rand <= 0)] = 0
        return rate
    
    def compute_interference(self, action):
        self.interference = np.zeros((self.numUsers, self.Antenna.bWNumb))+ self.channel.noise_pow
        intr = np.zeros((self.numUsers,1))
        
        for rx in range(self.numUsers):
            if not self.active_links[rx]:
                continue
            for beamInd in range(self.Antenna.bWNumb):
                for tx in range(self.numUsers):
                    if tx == rx:
                        intr[tx] = 0
                    elif tx!=rx and self.active_links[tx]:
                        Gtx,Grx = self.Antenna.AntGain (self.D2Dlinks[tx].txLoc, self.D2Dlinks[rx].rxLoc, self.Antenna.beamwidth[action[tx]], self.Antenna.beamwidth[beamInd], self.D2Dlinks[tx].txAntAng, self.D2Dlinks[rx].rxAntAng)
                        intr[tx] = self.power*self.ch_mtx[rx,tx]*Gtx*Grx*self.channel.fsPL*self.distan_mat[rx,tx]**(-self.channel.alpha)
                self.interference[rx,beamInd] = np.sum(intr) 

                        
        self.interference = 10*np.log10(self.interference)
    
    def act_for_training(self, actions):

        action_temp = actions.copy()
        rate, reward = self.compute_reward(action_temp)
        reward = np.sum(reward) / self.numUsers

        return reward
    
    def act_for_testing(self, actions):

        action_temp = actions.copy()
        rate,reward = self.compute_reward(action_temp)
        D2D_success = 1 - np.sum(self.active_links) / (self.numUsers)  # V2V success rates

        return rate, D2D_success
    
    def act_for_testing_rand(self, actions):

        action_temp = actions.copy()
        D2Drate = self.compute_reward_test(action_temp)
        D2D_success = 1 - np.sum(self.active_links_rand) / (self.numUsers)  # V2V success rates

        return D2Drate, D2D_success
    
    def new_game(self, numUsers = 0):
        
        if numUsers > 0:
            self.numUsers = numUsers
        self.add_new_link()
        self.renew_channel()
        # action0 = self.Antenna.beamwidth[np.zeros((self.numUsers,1),np.int8)] #### initialize all the antenna at 0
        # self.compute_link_timing(action0)
        self.data = self.data_size * np.ones((self.numUsers,1))
        self.individual_time_limit = self.horizon * np.ones((self.numUsers,1))
        self.active_links = np.ones((self.numUsers,1),dtype ='bool')
        
        self.data_rand = self.data_size * np.ones((self.numUsers,1))
        self.individual_time_limit_rand = self.horizon * np.ones((self.numUsers,1))
        self.active_links_rand = np.ones((self.numUsers,1),dtype ='bool')
        
# A = Environmnet(5,200,200,3,5,30,70)    
# A.new_game()
# ind = np.random.randint(0,10,(1,5))
# f = Antenna()
# action = f.beamwidth[ind]
# A.compute_reward(action)