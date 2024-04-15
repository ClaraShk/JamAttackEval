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


def create_reputation_df(path):
    reputationDf = pd.read_json(path)
    reputationDf = reputationDf['htlcs'].apply(pd.Series)
    expand = reputationDf['incomingCircuit'].apply(pd.Series)
    expand.rename(columns={'shortChannelId': 'shortChannelId_incoming', 'htlcIndex': 'htlcIndex_incoming'},
                  inplace=True)
    reputationDf = pd.concat([reputationDf, expand], axis=1)
    return reputationDf


def create_rep_by_channel_df(reputation_df):
    # Find unique values in 'shortChannelId_incoming'
    unique_channels = reputation_df['shortChannelId_incoming'].unique()

    # Create a new DataFrame with 'event time' and each unique channel as a column
    # Start by creating a dictionary with 'event time' as the key and an empty list as its value
    # Then, add each unique channel as a key with an empty list as its value
    data = {'event time': []}
    for channel in unique_channels:
        data[channel] = []

    # Create the DataFrame
    column_per_channel_df = pd.DataFrame(data)

    return column_per_channel_df


#After adding approx rep
def fill_reputation_by_channel(channel_df, reputation_df, reputation_cutoff = 0.05):
    #column_per_channel_df.sort_values(by='forwardTsNs') #probably don't need this

    reputation_df['approx rep'] = (reputation_df['incomingRevenue'] - reputation_df['outgoingRevenue'])

    new_df_with_events = pd.DataFrame({
        'event time': reputation_df['forwardTsNs'].values
    })

    # For every unique 'shortChannelId_incoming', add a column with NaN or another placeholder
    for channel in reputation_df['shortChannelId_incoming'].unique():  # change to pairs of channels
        new_df_with_events[channel] = np.nan

    for time in new_df_with_events['event time']:
        # Find rows in strong_reputationDf_short where 'forwardTsNs' matches 'time' and 'approx rep' is greater than 0
        condition_good = (reputation_df['forwardTsNs'] == time) & (
                    reputation_df['approx rep'] >= reputation_cutoff)
        filtered_df_good = reputation_df[condition_good]

        condition_low = (reputation_df['forwardTsNs'] == time) & (
                    reputation_df['approx rep'] < reputation_cutoff)
        filtered_df_low = reputation_df[condition_low]

        if not filtered_df_low.empty:
            # Assuming there's only one such match per 'time', get the 'shortChannelId_incoming' for that match
            incoming_channel = filtered_df_low['shortChannelId_incoming'].iloc[0]

            # Find the index(es) in new_df_with_events where 'event time' matches 'time'
            # Then use .loc to safely assign 'good' to 'incoming_channel' column for those index(es)
            indices = new_df_with_events[new_df_with_events['event time'] == time].index
            new_df_with_events.loc[indices, incoming_channel] = 'low'
        if not filtered_df_good.empty:
            # Assuming there's only one such match per 'time', get the 'shortChannelId_incoming' for that match
            incoming_channel = filtered_df_good['shortChannelId_incoming'].iloc[0]

            # Find the index(es) in new_df_with_events where 'event time' matches 'time'
            # Then use .loc to safely assign 'good' to 'incoming_channel' column for those index(es)
            indices = new_df_with_events[new_df_with_events['event time'] == time].index
            new_df_with_events.loc[indices, incoming_channel] = 'good'

    new_df_with_events.ffill(inplace=True)
    new_df_with_events['high rep neighbor'] = new_df_with_events.apply(
        lambda row: 'no' if not 'good' in row.values else 'yes', axis=1)
    return new_df_with_events


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


def updateStrongJamOp1(pair_schedule_df, channelsDf, inChannel, outChannel, numOfSlots):
    magic_jamming_number = 0.95
    balance_in = channelsDf[channelsDf['chan_id'] == inChannel]['capacity'].iloc[0].astype(int) * (1 / 2)  # balance in channel
    balance_out = channelsDf[channelsDf['chan_id'] == outChannel]['capacity'].iloc[0].astype(int) * (1 / 2)  # balance in channel
    locked_liq_in = 0
    locked_liq_out = 0
    available_slots_in = numOfSlots
    available_slots_out = numOfSlots
    #print(tabulate(channelsDf.head(), headers='keys'))
    for index, row in pair_schedule_df.iterrows():
        if (row['eventType'] == 'add') and (row['outgoingEndorsed'] == False):
            balance_in = balance_in - int(row['incomingAmount'])
            locked_liq_in = locked_liq_in + int(row['incomingAmount'])
            available_slots_in = available_slots_in - 1
        elif row['eventType'] == 'resolve':
            locked_liq_in = locked_liq_in - int(row['incomingAmount'])
            available_slots_in = available_slots_in + 1

        pair_schedule_df.at[index, 'balance in'] = balance_in
        pair_schedule_df.at[index, 'locked liq in'] = locked_liq_in

        if (locked_liq_in >= balance_in * magic_jamming_number) or (available_slots_in == 0):
            pair_schedule_df.at[index, 'strong jammed op1'] = True
        else:
            pair_schedule_df.at[index, 'strong jammed op1'] = False
    return pair_schedule_df


if __name__ == '__main__':
    in_channel = '273778395381760'
    out_channel = '267181325615104'
    slots_in_channel = 10
    forward_df = create_forwards_df('../BData/forwards.json')
    channels_df = create_channels_df('../BData/channels.json')
    reputation_df = create_reputation_df('../BData/reputation.json')
    pair_schedule_df = createJamScheduleForPair(in_channel, out_channel, forward_df) #Two channels in a given direction
#    print(tabulate(forward_df.head(30), headers='keys'))


    with_weak_jam = updateWeakJam(pair_schedule_df, channels_df, in_channel, out_channel, slots_in_channel)
    with_strong_jam_op1 = updateStrongJamOp1(pair_schedule_df, channels_df, in_channel, out_channel, slots_in_channel)
#    print(with_weak_jam['weak jammed'].head(10))
    print(tabulate(with_strong_jam_op1.head(30), headers='keys'))

    week_jam_by_neighbor_df = fill_reputation_by_channel(channels_df, reputation_df)
    print(tabulate(week_jam_by_neighbor_df.head(30), headers='keys'))





