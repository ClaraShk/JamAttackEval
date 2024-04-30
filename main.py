import pandas as pd
from tabulate import tabulate
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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

    #print(f'columns are {reputation_df.columns}')
    #print(tabulate(reputation_df.head(30), headers="keys"))


    # For every unique 'shortChannelId_incoming', add a column with NaN or another placeholder
    for channelIn in reputation_df['shortChannelId_incoming'].unique(): # change to pairs of channels
        for channelOut in reputation_df['outgoingChannel'].unique():
            new_df_with_events[channelIn+','+channelOut] = np.nan

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
            outgoing_channel = filtered_df_low['outgoingChannel'].iloc[0]
            #print(f'incoming channel {incoming_channel}, out {outgoing_channel}')

            # Find the index(es) in new_df_with_events where 'event time' matches 'time'
            # Then use .loc to safely assign 'good' to 'incoming_channel' column for those index(es)
            indices = new_df_with_events[new_df_with_events['event time'] == time].index
            new_df_with_events.loc[indices, incoming_channel+','+outgoing_channel] = 'low'
        if not filtered_df_good.empty:
            # Assuming there's only one such match per 'time', get the 'shortChannelId_incoming' for that match
            incoming_channel = filtered_df_good['shortChannelId_incoming'].iloc[0]
            outgoing_channel = filtered_df_good['outgoingChannel'].iloc[0]

            # Find the index(es) in new_df_with_events where 'event time' matches 'time'
            # Then use .loc to safely assign 'good' to 'incoming_channel' column for those index(es)
            indices = new_df_with_events[new_df_with_events['event time'] == time].index
            new_df_with_events.loc[indices, incoming_channel+','+outgoing_channel] = 'good'

    if False:
        new_row = pd.DataFrame({column: ['low'] if column != 'event time' else [None] for column in new_df_with_events.columns})
        new_row['event time'] = '0'

        new_df_with_events = pd.concat([new_row, new_df_with_events], ignore_index=True)

    new_df_with_events.ffill(inplace=True)
    new_df_with_events['high rep pair'] = new_df_with_events.apply(
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
    #filtered_new_df['approx outgoing rep'] = ''
    #filtered_new_df['strong jammed'] = ''

    # Order by 'timeOfEvent'
    ordered_df = filtered_new_df.sort_values(by='timeOfEvent')
    ordered_df = ordered_df.reset_index(drop = True)
#    print(f'in ordered_df for {inChannel} and {OutChannel} we have {len(ordered_df)} lines')
    #print(tabulate(ordered_df.head(30), headers='keys'))
    return ordered_df


# In a df of events, update the column 'weak jam'
# Currently only checking if slots and out is jammed
def updateWeakJam(eventDf, numOfSlots):

    magic_jamming_number = 0.95

    #eventDf['balance in'] = ''

    for channel in channels_df['chan_id'].unique():
        balance_in = channels_df[channels_df['chan_id'] == channel]['capacity'].iloc[0] * (1 / 2) * (
                    1 / 2)  # balance in general
        balance_out = channels_df[channels_df['chan_id'] == channel]['capacity'].iloc[0] * (1 / 2) * (
                    1 / 2)  # balance in general
        locked_liq_in = 0
        locked_liq_out = 0
        available_slots_in = numOfSlots/2 #slots in general
        available_slots_out = numOfSlots/2 #slots in general

        for index, row in eventDf[eventDf['shortChannelId_outgoing'] == channel].iterrows():
            if (row['eventType'] == 'add'): #and (row['outgoingEndorsed'] == False):
                #balance_in = balance_in - int(row['incomingAmount'])
                #locked_liq_in = locked_liq_in + int(row['outgoingAmount'])
                available_slots_out = available_slots_out - 1
            elif row['eventType'] == 'resolve':
               # locked_liq_in = locked_liq_in - int(row['incomingAmount'])
                available_slots_out = available_slots_out + 1

            #eventDf.at[index, 'balance in'] = balance_in
            #eventDf.at[index, 'locked liq in'] = locked_liq_in
            eventDf.at[index, 'taken slots out'] = available_slots_out



            if  (available_slots_out == 0): #locked_liq_in >= balance_in * magic_jamming_number) or
                eventDf.at[index, 'weak jammed'] = True
            else:
                eventDf.at[index, 'weak jammed'] = False
    return eventDf


# Currently only checking if slots and out is jammed
def updateStrongJamOp1(eventDf, channelsDf, numOfSlots):
    magic_jamming_number = 0.95

    for channel in channels_df['chan_id'].unique():
        balance_in = channelsDf[channelsDf['chan_id'] == channel]['capacity'].iloc[0].astype(int) * (1 / 2)  # balance in channel
        #balance_out = channelsDf[channelsDf['chan_id'] == outChannel]['capacity'].iloc[0].astype(int) * (1 / 2)  # balance in channel
        locked_liq_in = 0
        locked_liq_out = 0
        available_slots_in = numOfSlots
        available_slots_out = numOfSlots
        #print(tabulate(channelsDf.head(), headers='keys'))
        for index, row in eventDf[eventDf['shortChannelId_outgoing'] == channel].iterrows():

            if (row['eventType'] == 'add'):  # and (row['outgoingEndorsed'] == False):
                # balance_in = balance_in - int(row['incomingAmount'])
                # locked_liq_in = locked_liq_in + int(row['outgoingAmount'])
                numOfSlots = numOfSlots - 1
            elif row['eventType'] == 'resolve':
                # locked_liq_in = locked_liq_in - int(row['incomingAmount'])
                numOfSlots = numOfSlots + 1
            #pair_schedule_df.at[index, 'balance in'] = balance_in
            #pair_schedule_df.at[index, 'locked liq in'] = locked_liq_in
            pair_schedule_df.at[index, 'available slots'] = numOfSlots

            if (numOfSlots == 0): #locked_liq_in >= balance_in * magic_jamming_number) or
                pair_schedule_df.at[index, 'strong jammed op1'] = True
            else:
                pair_schedule_df.at[index, 'strong jammed op1'] = False
    return pair_schedule_df

def times_of_change(jamDf):
    changes = jamDf['strong jam'] != jamDf['strong jam'].shift(1)
    indices_of_changes = changes[changes].index
    indices_to_keep = sorted(
        set(indices_of_changes.union(indices_of_changes - 1).intersection(range(len(jamDf)))))
    change_table = jamDf.loc[indices_to_keep]
    return change_table

def calc_total_jam():
    # Detect changes in 'strong_jam'
    with_op2['change'] = with_op2['strong jam'] != with_op2['strong jam'].shift()

    # Identify start and end indices of True periods
    starts = with_op2[(with_op2['strong jam'] == True) & (with_op2['change'] == True)].index
    ends = with_op2[(with_op2['strong jam'] == True) & (with_op2['change'].shift(-1) == True)].index

    # Prepare a list to hold durations
    true_durations = []

    # Iterate over starts and ends to calculate durations
    for start, end in zip(starts, ends):
        start_time = with_op2.loc[start, 'time']
        end_time = with_op2.loc[end + 1, 'time'] if end + 1 < len(with_op2) else with_op2.loc[end, 'time']
        duration = int(end_time) - int(start_time)
        true_durations.append(duration)

    # Compute the total time
    total_true_time = sum(true_durations)

    # Output the results
    #print(f"True periods and their durations: {true_durations}")
    print(f"Total time of strong jam in nanoseconds: {total_true_time}")


def calc_fees():
    forward_df['fee'] = forward_df['incomingAmount'].astype(int) - forward_df['outgoingAmount'].astype(int)
    return forward_df['fee'].sum()


def updateStrongJamOp2(rep_by_channel_df, events_df):
    df1 = rep_by_channel_df[['event time', 'high rep pair']].copy()
    df2 = events_df[['timeOfEvent', 'weak jammed', 'strong jammed op1']].copy()
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df['time'] = np.where(combined_df['event time'].isna(), combined_df['timeOfEvent'],
                                   combined_df['event time'])
    combined_df = combined_df.sort_values(by='time')

    clean_df = combined_df[['time', 'high rep pair', 'weak jammed', 'strong jammed op1']].sort_values(
        by='time').copy()
    clean_df.ffill(inplace=True)
    clean_df['strong jam op2'] = np.where(((clean_df['high rep pair'] == 'no') & (clean_df['weak jammed'] == True)),
                                          True, False)
    clean_df['strong jam'] = np.where(((clean_df['strong jammed op1']) | (clean_df['strong jam op2'])), True, False)
    return clean_df.reset_index(drop = True)


def unpack_htlcs(htlcs):
    # Unpack 'htlcs' if not empty and is a list
    if htlcs and isinstance(htlcs, list):
        # List comprehensions to extract data from each htlc, including nested 'route'
        status = [htlc.get('status') for htlc in htlcs]
        #timestamps = [htlc.get('timestamp') for htlc in htlcs]
        # Extract 'total_fees_msat' from the 'route' within each 'htlcs' if it exists
        total_fees_msat = [htlc.get('route', {}).get('total_fees_msat') if htlc.get('route') else None for htlc in htlcs]

        return pd.Series({
            'status': status[0],
            'total_fees_msat': int(total_fees_msat[0])
        })


def extract_data(payment):
    # Extract data from both 'htlcs' and 'route'
    data = {}
    if 'htlcs' in payment:
        data.update(unpack_htlcs(payment['htlcs']))
    return pd.Series(data)

if __name__ == '__main__':
    #in_channel = '273778395381760'
    #out_channel = '267181325615104'
    slots_in_channel = 2


    forward_df = create_forwards_df('files/forwarding_history.json')
    channels_df = create_channels_df('files/channels.json')
    reputation_df = create_reputation_df('files/reputation_thresholds.json')

    paymentsDf = pd.read_json('files/payments.json')

   #find all pairs and go over them
    if True:
        results_dfs = []
        for in_channel in channels_df['chan_id'].unique():
            for out_channel in channels_df['chan_id'].unique():
                if in_channel == out_channel:
                    continue
                else:
                    results_dfs.append(createJamScheduleForPair(in_channel, out_channel, forward_df)) #Two channels in a given direction
#                    print(tabulate(pair_schedule_df.head(30), headers='keys'))
        concat_df = pd.concat(results_dfs, ignore_index=True)
        ordered_df = concat_df.sort_values(by='timeOfEvent')
        pair_schedule_df = ordered_df.reset_index(drop = True)




    else:
        in_channel = '273778395381760'
        out_channel = '267181325615104'
        pair_schedule_df = createJamScheduleForPair(in_channel, out_channel, forward_df)

#    print(tabulate(forward_df.head(30), headers='keys'))

    with_weak_jam = updateWeakJam(pair_schedule_df,  slots_in_channel)
    #print('pair_schedule_df:')
    #print(tabulate(pair_schedule_df.head(10), headers='keys'))

#    print(tabulate(with_weak_jam.head(30), headers='keys'))
    with_strong_jam_op1 = updateStrongJamOp1(pair_schedule_df, channels_df,  slots_in_channel)
#    print(with_weak_jam['weak jammed'].head(10))
    #print('OP1 update:')
    #print(tabulate(with_strong_jam_op1.head(30), headers='keys'))

    week_jam_by_neighbor_df = fill_reputation_by_channel(channels_df, reputation_df)
    #print('weak jam by n')
    #print(tabulate(week_jam_by_neighbor_df.head(30), headers='keys'))

    with_op2 = updateStrongJamOp2(week_jam_by_neighbor_df, with_strong_jam_op1)

    changes = times_of_change(with_op2)
    #print('changes:')
    #print(tabulate(changes.head(30), headers='keys'))

    paymentsDetaildf = paymentsDf['payments'].apply(extract_data)

    #print(tabulate(with_op2, headers='keys'))
    #print('times of strong jam')
    #print(tabulate(with_op2[with_op2['strong jam']==True].head(30), headers='keys'))
    #print(tabulate(with_op2[with_op2["weak jammed"]==True].head(30), headers='keys'))
    #print(f"num of events {len(with_op2)}")
    #print(f"weak jam {len(with_op2[with_op2['weak jammed']==True])}")
    #print(f"strong jam {len(with_op2[with_op2['strong jam']==True])}")
    print(f"HTLCs sent {len(pair_schedule_df[pair_schedule_df['eventType'] == 'add'])}")
    #print(f'Fees: {calc_fees()}')

    calc_total_jam()

    success = paymentsDetaildf[paymentsDetaildf['status'] == 'SUCCEEDED']['total_fees_msat'].sum()
    upfront = paymentsDetaildf['total_fees_msat'].sum() * 0.01
    print(f'Success: {success}')
    print(f'Upfront: {upfront}')




    #print(f'this is {with_op2[with_op2["time"] == "1712783765395927992"]["high rep pair"]}')







