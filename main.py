import pandas as pd
from tabulate import tabulate
import numpy as np


#read and parse forwards json
def create_forwards_df(path):
    forwardsDf = pd.read_json(path)
    forwardsDf = forwardsDf['forwards'].apply(pd.Series)
    expand_in = forwardsDf['incomingCircuit'].apply(pd.Series)
    expand_in.rename(columns={'shortChannelId': 'shortChannelId_incoming', 'htlcIndex': 'htlcIndex_incoming'},
                     inplace=True)
    forwardsDf = pd.concat([forwardsDf, expand_in], axis=1)
    expand_out = forwardsDf['outgoingCircuit'].apply(pd.Series)
    expand_out.rename(columns={'shortChannelId': 'shortChannelId_outgoing', 'htlcIndex': 'htlcIndex_outgoing'},
                      inplace=True)
    forwardsDf = pd.concat([forwardsDf, expand_out], axis=1)

    return forwardsDf

#read and parse channels json
def create_channels_df(path):
    channelsDf = pd.read_json(path)
    channelsDf = channelsDf['channels'].apply(pd.Series)
    channelsDf['capacity'] = channelsDf['capacity'].astype(int)
    return channelsDf

#For a pair of channels, create a df in which each row correponds to a time of interest (htlc added/resolved)
def createJamScheduleForPair(inChannel, OutChannel, forwardsDf):
    # Filter rows where shortChannelId_incoming matches inputChannel
    #pair_df = forwardsDf[(forwardsDf['shortChannelId_incoming'] == inChannel) & (forwardsDf['shortChannelId_outgoing'] == OutChannel)].copy()

    # Splitting the original DataFrame into two, one for each time event
    add_df = forwardsDf.copy()
    add_df['timeOfEvent'] = add_df['addTimeNs']
    add_df['eventType'] = 'add'
    #add_df.drop(['addTimeNs'], axis=1, inplace=True)

    resolve_df = forwardsDf.copy()
    resolve_df['timeOfEvent'] = resolve_df['resolveTimeNs']
    resolve_df['eventType'] = 'resolve'
    #resolve_df.drop(['resolveTimeNs'], axis=1, inplace=True)

    # Concatenate the two DataFrames
    new_df = pd.concat([add_df, resolve_df])

    # Filter rows where shortChannelId_incoming matches inputChannel
    filtered_new_df = new_df[(new_df['shortChannelId_incoming'] == inChannel) & (new_df['shortChannelId_outgoing'] == OutChannel)].copy()

    # Add empty columns for 'weak jammed' and 'strong jammed'
    filtered_new_df['weak jammed'] = ''
    filtered_new_df['approx outgoing rep'] = ''
    filtered_new_df['strong jammed'] = ''

    # Order by 'timeOfEvent'
    ordered_df = filtered_new_df.sort_values(by='timeOfEvent')
    ordered_df = ordered_df.reset_index()
    return ordered_df

#In a df of events, update the column 'weak jam'
def updateWeakJam(eventDf, channelsDf, inChannel, outChannel, numOfSlots): #Do we care about balance out? Probably...
    magic_jamming_number = 0.95
    balance_in = channelsDf[channelsDf['chan_id'] == inChannel]['capacity'][0] * (1 / 2) * (
                1 / 2)  # balance in general
    balance_out = channelsDf[channelsDf['chan_id'] == outChannel]['capacity'] * (1 / 2) * (
                1 / 2)  # balance in general
    locked_liq_in = 0
    locked_liq_out = 0
    available_slots_in = numOfSlots
    available_slots_out = numOfSlots
    eventDf['balance in'] = ''
    for index, row in eventDf.iterrows():
        if (row['eventType'] == 'add') and (row['outgoingEndorsed'] == False):
            balance_in = balance_in - int(row['incomingAmount'])
            locked_liq_in = locked_liq_in + int(row['incomingAmount'])
            available_slots_in = available_slots_in - 1
        elif row['eventType'] == 'resolve':
            locked_liq_in = locked_liq_in - int(row['incomingAmount'])
            available_slots_in = available_slots_in + 1

        eventDf.at[index, 'balance in'] = balance_in
        eventDf.at[index, 'locked liq in'] = locked_liq_in



        if (locked_liq_in >= balance_in * magic_jamming_number) or (available_slots_in == 0):
            eventDf.at[index, 'weak jammed'] = True
        else:
            eventDf.at[index, 'weak jammed'] = False
    return eventDf


if __name__ == '__main__':
    in_channel = '273778395381760'
    out_channel = '267181325615104'
    slots_in_channel = 10
    forward_df = create_forwards_df('../BData/forwards.json')
    channels_df = create_channels_df('../BData/channels.json')
    pair_schedule_df = createJamScheduleForPair(in_channel, out_channel, forward_df)
    with_weak_jam = updateWeakJam(pair_schedule_df, channels_df, in_channel, out_channel, slots_in_channel)
#    print(with_weak_jam['weak jammed'].head(10))
    print(tabulate(with_weak_jam.head(10), headers='keys'))


