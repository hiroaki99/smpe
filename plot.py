import json 
import numpy as np
import matplotlib.pyplot as plt 

def plot(filename, legend):
    res = open(filename + "metrics.json")
    res = json.load(res)
    plt.plot( res["test_return_mean"]["values"], label=legend)


# traffic
plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/traffic_junction_medium/11/", "maa2c")
plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/papou_vae_maa2c/traffic_junction_medium/1/", "CAM")


# # Foraging-5x5-3p-2f-coop-v2
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/Foraging-5x5-3p-2f-coop-v2/3/", "maa2c")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/Foraging-5x5-3p-2f-coop-v2/4/", "maa2c")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/ATM/results/sacred/lbforaging:Foraging-6x6-3p-2f-coop-v2/maa2c/3/", "ATM")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/Foraging-5x5-3p-2f-coop-v2/19/", "Centralized maa2c (64)")
# plt.title("lbforaging:Foraging-5x5-3p-2f-coop-v2")


# Foraging-6x6-3p-2f-coop-v2
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/Foraging-6x6-3p-2f-coop-v2/1/", "maa2c")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/Foraging-6x6-3p-2f-coop-v2/2/", "maa2c")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/ATM/results/sacred/lbforaging:Foraging-6x6-3p-2f-coop-v2/maa2c/3/", "ATM")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/Foraging-6x6-3p-2f-coop-v2/23/", "Centralized maa2c (64)")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/Foraging-6x6-3p-2f-coop-v2/24/", "Centralized maa2c (64)")
# plt.title("lbforaging:Foraging-6x6-3p-2f-coop-v2")


# # Foraging-7x7-3p-2f-coop-v2
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/Foraging-7x7-3p-2f-coop-v2/1/", "maa2c")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/Foraging-7x7-3p-2f-coop-v2/2/", "maa2c")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/ATM/results/sacred/lbforaging:Foraging-7x7-3p-2f-coop-v2/maa2c/3/", "ATM")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/Foraging-7x7-3p-2f-coop-v2/1/", "Centralized maa2c (64)")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/Foraging-7x7-3p-2f-coop-v2/24/", "Centralized maa2c (64)")
# plt.title("lbforaging:Foraging-7x7-3p-2f-coop-v2")


# Foraging-15x15-3p-5f-v2
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/Foraging-15x15-3p-5f-v2/3/", "maa2c")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/ATM/results/sacred/lbforaging:Foraging-15x15-3p-5f-v2/maa2c/4/", "ATM")
# plt.title("lbforaging:Foraging-15x15-3p-5f-v2")

# mpe:SimpleSpeakerListener-v0
plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/mpe:SimpleSpeakerListener-v0/3/", "maa2c")
plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/mpe:SimpleSpeakerListener-v0/1/", "Centralized maa2c")
plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/mpe:SimpleSpeakerListener-v0/3/", "Centralized maa2c")
plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/dynamics_maa2c/mpe:SimpleSpeakerListener-v0/12/", "Dynamics maa2c")
plt.title("SimpleSpeakerListener")

# # mpe:SimpleSpread-v0
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/maa2c/mpe:SimpleSpread-v0/2/", "maa2c")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/ATM/results/sacred/lbforaging:Foraging-6x6-3p-2f-coop-v2/maa2c/3/", "ATM")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/mpe:SimpleSpread-v0/5/", "Centralized maa2c")
# plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/dynamics_maa2c/mpe:SimpleSpread-v0/43/", "Dynamics maa2c")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/mpe:SimpleSpread-v0/6/", "Centralized maa2c (new)")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/mpe:SimpleSpread-v0/7/", "Centralized maa2c (old)")
# # plot("/home/ddaedalus/Documents/files/EPyMARL/epymarl-main/results/sacred/centralized_maa2c/mpe:SimpleSpread-v0/8/", "Centralized maa2c (new)")
# plt.title("SimpleSpread")


plt.legend()
plt.show()
	
	