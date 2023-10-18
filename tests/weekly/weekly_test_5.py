df = pd.read_csv("C:/Users/MSI laptop/Downloads/chipotle.tsv", sep='\t')

def change_price_to_float(input_df):
    df2 = input_df.copy()
    df2["item_price"] = input_df["item_price"].str.replace("$", "", regex = False).astype(float)
    return df2

def number_of_observations(input_df):
    return len(input_df)

def items_and_prices(input_df):
    list_ = ["item_name", "item_price"]
    return  df[list_].drop_duplicates()


def sorted_by_price(input_df):
    list_ = ["item_name", "item_price"]
    df2 = df[list_].drop_duplicates()
    return df2.sort_values("item_price", ascending = False)


def avg_price(input_df):
    return input_df["item_price"].mean()


def unique_items_over_ten_dollars(input_df):
    list_ = ["item_name", "item_price", "choice_description"]
    df2 = input_df[list_].drop_duplicates()
    df2 = df2[df2["item_price"] > 10]
    return df2["item_name"]


def items_starting_with_s(input_df):
    list_ = []
    for i in range(len(input_df)):
        if input_df["item_name"][i].startswith("S") == True:
            list_.append(df["item_name"][i])
    series = pd.DataFrame(list_).squeeze()
    series.name="item_name"
    return series


def first_three_columns(input_df):
    return df.iloc[:, 0:3]



def every_column_except_last_two(input_df):
    return df.iloc[:, 0:len(input_df.columns)-2]



def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    input_df2 = input_df.copy()
    input_df3 = input_df2.loc[df[column_to_filter].isin(rows_to_keep),columns_to_keep]
    return input_df3



def generate_quartile(input_df):
    input_df2 = input_df.copy()
    input_df2["Quartile"] = 0
    for i in range(len(input_df2)):
        if input_df2["item_price"][i] >= 30:
            input_df2["Quartile"][i] = "premium"
        else:
            if 20 <= input_df2["item_price"][i] < 30:
                input_df2["Quartile"][i] = "high-cost"
            else:
                if 10 <= input_df2["item_price"][i] < 20:
                    input_df2["Quartile"][i] = "medium-cost"
                else:
                    input_df2["Quartile"][i] = "low-cost"

    return input_df2



def average_price_in_quartiles(input_df):
    df2 = generate_quartile(input_df)
    dummy_list = ['item_price']
    return df2.groupby('Quartile')[dummy_list].mean().reset_index()




def minmaxmean_price_in_quartile(input_df):
    input_df2 = input_df.copy()
    input_df3 = input_df2.groupby('Quartile').agg({'item_price': ['min', 'max']}).reset_index()
    return input_df3





def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for i in range(number_of_trajectories):
        trajectory_mean = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(distribution.gen_rand())
            trajectory_mean.append(sum(trajectory)/(i+1))
        trajectories.append(trajectory_mean)

    return trajectories