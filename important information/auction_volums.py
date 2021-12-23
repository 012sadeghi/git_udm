import pandas
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

import os

# plt.close(all)
# TODO: Solar Forecast
start1 = 3
end1 = 3 + 24
start2, end2 = 286972, 287020
start3, end3 = 287050, 287121
start4, end4 = 287159, 287219

start, end = start1, end1
# Load data
df = pandas.read_excel("MarketData/xls_data.xls")

# # start, end = start4, end4
# # TODO: Change the format of data to make it compatible with what we want ..
auction_volume_buy = df['Buy']
# print(auction_volume_buy[0])
# b = auction_volume_buy[3]
# c = b.replace(',', '')
# print(float(c))
# TODO: Print auction data pdf
# print(auction_volume_buy)
auction_volume_buy[24*80:24*80+24].plot.kde()
gm_data = auction_volume_buy[24*80:24*80+24]
# kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
# kde.fit(gm_data)


# plt.xlabel('Buy [kWh] (auction volums)')
# plt.title('Density')
# plt.grid()
# plt.show()

# gm = GaussianMixture(n_components=2).fit(np.array(gm_data.reshape(-1, 1)))

class auction_data_generation(object):
    def __init__(self, data=df):
        self.data = data
        self.day = 0
        self.counter = 0
        self.sampled_date = ''
        self.sample_4_days()

    def sample_4_days(self):
        self.days = np.random.randint(low=1, high=364, size=4)
    # def sample_a_day(self):
    #     self.year = np.random.randint(low=2014, high=2020, size=1)
    #     self.month = np.random.randint(low=1, high=12, size=1)
    #     self.day = np.random.randint(low=1, high=12, size=1)
    #     self.sampled_date = str(self.year[0])+'-'+'%02d' %self.month[0]+'-'+'%02d' %self.day[0]+' '+'15:00:00'
    #     self.episode_index = df.index[df['DateTime'] == self.sampled_date]
    #     if len(self.episode_index) == 0: self.episode_index = [np.random.randint(low=98322, high=108322)]
        # d = df['DateTime']
        # self.episode_index = d[self.sampled_date]
        # self.episode_index = df['DateTime'][self.sampled_date]

    def auction_sample(self, t):
        t = t % 56
        if t == 0: self.sample_4_days()
        self.counter = (self.counter + 1) % 4
        ret = self.data['Buy'][self.days[self.counter] * 24 + t]

        return ret / 100000











# c = (b.astype(float))
# print(e)
# plt.figure()
# auction_volume_buy.plot()
# # x = np.sort(auction_volume_buy[start:end])
# # y = np.arange(np.size(auction_volume_buy[start:end]))/np.size(auction_volume_buy[start:end])
# # print(np.max(df['auction_volume_buy'][start:end]))
# # ax0 = plt.plot(range(start, end), df['Unnamed: 3'][start:end])
# plt.show()
#
# # d_realtime = df['Real-time Upscaled Measurement [MW]']
# # x = np.sort(d_realtime[start:end])
# # y = np.arange(np.size(d_realtime[start:end]))/np.size(d_realtime[start:end])
# # ax1 = plt.plot(x, y)
# #
# # d_mostrecent = df['Most recent forecast [MW]']
# # x = np.sort(d_mostrecent[start:end])
# # y = np.arange(np.size(d_mostrecent[start:end]))/np.size(d_mostrecent[start:end])
# # ax2 = plt.plot(x, y)
# #
# #
# # d_dayahead = df['Day-Ahead forecast [MW]']
# # x = np.sort(d_dayahead[start:end])
# # y = np.arange(np.size(d_dayahead[start:end]))/np.size(d_dayahead[start:end])
# # ax3 = plt.plot(x, y)
# #
# # plt.show()
# d_weekahead = df['Week-Ahead forecast [MW]']
# # count, bins_count = np.histogram(d_weekahead, bins=10)
# # pdf = count / sum(count)
# # cdf = np.cumsum(pdf)
# # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
# # plt.plot(bins_count[1:], cdf, label="CDF")
# # plt.legend()
# x = np.sort(d_weekahead[start:end])
# y = np.arange(np.size(d_weekahead[start:end]))/np.size(d_weekahead[start:end])
# ax4 = plt.plot(x, y)
#
#
# plt.xlabel('Power')
# plt.ylabel('CDF')
# plt.ylim(-.01, 0.99999)
# plt.legend(['Real-time Measurement [MW]','Most recent forecast [MW]','Day-Ahead forecast [MW]','Week-Ahead forecast [MW]'])
# # plt.show()
#
# sample = df.sample(n=1)
# print("first sample", sample)
#
# ## PDFs
# ax2 = df['Real-time Upscaled Measurement [MW]'][start:end].plot.kde()
# ax2 = df['Most recent forecast [MW]'][start:end].plot.kde()
# ax2 = df['Day-Ahead forecast [MW]'][start:end].plot.kde()
# ax2 = df['Week-Ahead forecast [MW]'][start:end].plot.kde()
# plt.legend(['Real-time Measurement [MW]','Most recent forecast [MW]','Day-Ahead forecast [MW]','Week-Ahead forecast [MW]'])
#
#
# # plt.show()
#
# samples = df['Day-Ahead forecast [MW]'][start:end].sample(n=3)
# sample = samples.sample(n=1)
# print(samples)
# ax = df.plot(y="Real-time Upscaled Measurement [MW]", kind='line', xlim=(start, end))
# df.plot(ax=ax, y="Most recent forecast [MW]", kind='line', xlim=(start, end))
# df.plot(ax=ax, y="Day-Ahead forecast [MW]", kind='line', xlim=(start, end))
# df.plot(ax=ax, y="Week-Ahead forecast [MW]", kind='line', xlim=(start, end))
# df.plot(ax=ax, y="Day-Ahead forecast (11h00) [MW]", kind='line', xlim=(start, end))
#
# # plt.show()
# print(df['Real-time Upscaled Measurement [MW]'])
# print(df['DateTime'].str.find('2021-12-05 23:45:00'))
