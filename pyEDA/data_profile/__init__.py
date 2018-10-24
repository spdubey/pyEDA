import pandas as pd
from collections import OrderedDict
import os
from os import listdir
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import base64
import webbrowser
import re
import seaborn as sns
import sys
import warnings
import calendar
import missingno as msno

class ExploratoraryDataAnalysis:

    def __init__(self):
        self.clm_desc = {}
        self.HTML_content = []
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    def _read_data(self):
        """
       :return:
        Function reads the data from a csv file.

        Output:
            self.raw_data: read data
            file_size: Size of the read data
        """
        self.file_name = sys.argv[1]
        self.working_path_dir = os.getcwd()
        self.file_path = '{}{}'.format(self.working_path_dir, '\\')
        self.fle_loc = '{}{}{}'.format(self.file_path, '\\', self.file_name)
        self.file_size = os.path.getsize('{}{}{}'.format(self.file_path, '\\', self.file_name))
        if os.path.exists(self.file_path):
            if self.file_name.split(".")[1].lower() == 'csv':
                self.raw_data = pd.read_csv(self.fle_loc)
            else:
                self.raw_data = pd.read_excel(self.fle_loc)
        else:
            sys.exit("The file '{}' does not exists on path '{}'".format(self.file_name, self.file_path))

    def data_profiling(self, input_dataframe, input_file_name, target_variable, date_time_var, col_to_be_excluded):
        """
        Checks the data type of all the column and also converts the column to date time
        if it is in the format like dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy
        input: data read from csv or excel sheet
        Output:
            self.numeric_clms: List of all numeric columns
            self.categorical_clms: List of all categorical columns
            self.bool_clms: List of all boolean columns
            self.date_time_clms: List of all date time columns
            self.clm_desc: dictionary with column names and data types
        """
        self.col_to_be_excluded = col_to_be_excluded
        self.raw_data = input_dataframe
        self.file_name = input_file_name
        self.target_variable = target_variable
        self.date_time_var = date_time_var
        self.date_time_clms = []
        # checking if there is any column to be excluded in eda
        if self.col_to_be_excluded:
            if len(self.col_to_be_excluded) > 0:
                self.raw_data.drop(labels=self.col_to_be_excluded, axis=1, inplace=True)
        if self.date_time_var:
            if len(self.date_time_var) > 0:
                for clm in self.date_time_var:
                    if clm in self.raw_data.dtypes.index:
                        self.date_time_clms.append(clm.strip())
                    else:
                        print("The date time column '{}' you entered is not in data".format(self.date_time_var))
        for clm in self.raw_data.dtypes.index:
            if self.raw_data.dtypes[clm] == 'O':
                if len(self.raw_data[clm].unique()) == len(self.raw_data[clm]):
                    self.clm_desc[clm] = 'Text'
                elif self._date_time_check(clm) | (clm in self.date_time_clms):
                    self.raw_data[clm] = pd.to_datetime(self.raw_data[clm], dayfirst=True)
                    self.clm_desc[clm] = 'DateTime'
                else:
                    self.clm_desc[clm] = 'Object'
            elif (self.raw_data.dtypes[clm] == 'int64') | (self.raw_data.dtypes[clm] == 'int32'):
                if len(self.raw_data[clm].unique()) < 10:
                    self.clm_desc[clm] = 'Object'
                    self.raw_data[clm] = self.raw_data[clm].astype('category')
                else:
                    self.clm_desc[clm] = 'Integer'
            elif self.raw_data.dtypes[clm] == 'float64':
                self.clm_desc[clm] = 'Float'
            elif (self.raw_data.dtypes[clm] == 'datetime64[ns]') | (self.raw_data.dtypes[clm] == '<M8[ns]'):
                self.clm_desc[clm] = 'DateTime'
            elif self.raw_data.dtypes[clm] == 'bool':
                self.clm_desc[clm] = 'Object'
            else:
                self.clm_desc[clm] = self.raw_data.dtypes[clm]
        self.numeric_clms = [clm for clm, dtype in self.clm_desc.items() if dtype in ['Integer', 'Float']]
        self.categorical_clms = [clm for clm, dtype in self.clm_desc.items() if dtype in ['Object']]
        self.bool_clms = [clm for clm, dtype in self.clm_desc.items() if dtype in ['bool']]
        self.date_time_clms.extend([clm for clm, dtype in self.clm_desc.items() if dtype in ['DateTime']])
        self.date_time_clms = list(set(self.date_time_clms))

    def _date_time_check(self, clm):
        self.regex_list = self.regex_list = ['(\d+/\d+/\d+)', '(\d+-\d+-\d+)', '(\d+\.\d+\.\d+)']
        check = any(re.match(regex, self.raw_data[clm][~pd.isnull(self.raw_data[clm])][1]) for regex in self.regex_list)
        return check

    def _create_report_head(self):
        """Creating the Heading with number of rows and columns"""
        self.file_size = self.raw_data.memory_usage(index=True).sum()
        self.HTML_content.append('<center><h1 style="font-size:2.5vw;">Exploratory Data Analysis Report</h1></center>')
        text_cursor = '<h1 style="font-size:1.5vw;">Data Shape: Rows = {}, Columns = {}</h1>'.format(
            self.raw_data.shape[0], self.raw_data.shape[1])
        self.HTML_content.append(text_cursor)
        text_cursor = '<h1 style="font-size:1.5vw;">Data size in memory = {} MB</h1>'.format(
            round(self.file_size/1000000, 1))
        self.HTML_content.append(text_cursor)
        # Creating the sample data in HTML
        html_df = self._df_to_html_converter(self.raw_data.head(10))
        self._html_formatter(content=html_df, heading='Preview')

    def _text_adder(self, message):
        # text_cursor = '<ul style="font-size:0.8vw;">{}</ul>'.format(text)
        self.HTML_content.append(message)

    def _html_df_obj_creator(self, df, heading):
        # Creating HTML object of this df
        html_df = self._df_to_html_converter(df)
        self._html_formatter(content=html_df, heading=heading)

    def _html_image_obj_creator(self, img, heading):
        image_html = self._image_to_html_converter(img)
        self._html_formatter(content=image_html, heading=heading)

    def _df_to_html_converter(self, df):
        html_df = pd.DataFrame.to_html(df)
        return html_df

    def _image_to_html_converter(self, image_file_name):
        image_data = base64.b64encode(open(image_file_name, 'rb').read()).decode('utf-8').replace('\n', '')
        image_html = '<img src="data:image/png;base64,{}"s>'.format(image_data)
        return image_html

    def _html_formatter(self, content, heading):
        self.HTML_content.append('<br /><h1 style="font-size:1.5vw;">{}<br /></h1>'.format(heading))
        self.HTML_content.append(content)
        self.HTML_content.append('<br /><br /><br /><br />')

    def _html_writer(self):
        f = open('EDA_Report_{}.html'.format(self.file_name), 'w')
        f.write(''.join(self.HTML_content))
        f.close()
        webbrowser.open_new_tab('EDA_Report_{}.html'.format(self.file_name))

    def _create_data_summary(self):
        self.summary_df = self.raw_data[self.numeric_clms].describe()
        self.summary_df = self.summary_df.apply(lambda col: round(col, 2))
        self._html_df_obj_creator(self.summary_df, heading='Data Summary')

    def _miss_value_df_creator(self):
        # Creating Missing Value Data Frame
        temp_list = []
        for col in list(pd.isnull(self.raw_data).sum().index):
            temp_list.append(self.clm_desc[col])
        self.missing_value_table = pd.DataFrame({'Column Names': list(pd.isnull(self.raw_data).sum().index),
                                                 'Data Type': temp_list,
                                                 'Missing Value': list(pd.isnull(self.raw_data).sum())})
        self.missing_value_table['Missing Percentage'] = round(
            (self.missing_value_table['Missing Value']/len(self.raw_data))*100, 2)

    def _distinct_value_df_creator(self):
        uniqe_val = pd.DataFrame(self.raw_data.apply(lambda col: len(col.unique())))
        uniqe_val.reset_index(inplace=True)
        uniqe_val.columns = ['Column Names', 'Distinct Value']
        uniqe_val['Distinct Percentage'] = uniqe_val['Distinct Value'].apply(
            lambda x: round((x/len(self.raw_data)*100), 2))
        self.distinct_value_table = uniqe_val
        self.distinct_value_table.reset_index(drop=True, inplace=True)

    def _missing_distinct_plot(self):
        """
       :return:
        Function plots the graphical representation for each categorical columns in the dataframe.

        Input:
        object_x: pandas DataFrame
        size: vertical and horizontal size of the plot
        """
        plt.gcf().clear()
        x_pos = np.arange(len(self.missing_value_table)*8, step=8)
        plt.bar(x_pos, self.missing_value_table['Missing Percentage'], color='r', width=3, align='center')
        plt.bar(x_pos+1.5, self.distinct_value_table['Distinct Percentage'], color='#0504aa', width=3)
        plt.xticks(x_pos, self.missing_value_table['Column Names'], rotation=90)
        plt.ylabel('Percentage Value')
        plt.legend(['Missing %age', 'Distinct %age'], loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True,
                   ncol=1)
        image_file_name = '{}.png'.format('Missing'.replace(' ', ''))
        plt.savefig(image_file_name, bbox_inches='tight')
        self._html_image_obj_creator(image_file_name, heading='Missing and Distinct percentage plot')

    def _miss_distinct_df_creator(self):
        # Merging distinct and missing value table
        merged_df = pd.merge(self.missing_value_table, self.distinct_value_table)
        merged_df.index.names = ['S.No']
        self._html_df_obj_creator(merged_df, heading='Missing and Distinct Value Table')
        self.high_missval_df_warn = list(merged_df['Column Names'][merged_df['Missing Percentage'] > 10])
        # Adding warning for missing value
        if len(self.high_missval_df_warn) > 0:
            text_cursor = '<h1 style="font-size:1.2vw;">{}<br /></h1>'.format('Warnings : ')
            self.HTML_content.append(text_cursor)
            for i, row in merged_df.iterrows():
                if row['Missing Percentage'] > 10:
                    message = '<li>{} have high missing value( percentage = {})</li>'.format(
                        row['Column Names'], row['Missing Percentage'])
                    self.HTML_content.append(message)

    def _categorical_value_count_plot(self):
        if len(self.categorical_clms) != 0:
            self.HTML_content.append('<h1 style="font-size:1.5vw;">Categorical column:</h1>')
            ids_clm = []
            temp = []
            for clm in self.categorical_clms:
                if len(self.raw_data[clm].unique()) >= int(0.80 * len(self.raw_data[clm])):
                    ids_clm.append(clm)
            plt.gcf().clear()
            if len(self.categorical_clms) == 1:
                row = 1
                fig, axs = plt.subplots(row, figsize=(6, 6), facecolor='w', edgecolor='k')
                for i in range(row):
                    sns.countplot(self.categorical_clms[i], data=self.raw_data, ax=axs)
                    if len(self.raw_data[self.categorical_clms[i]].unique()) < 20:
                        axs.set_xticklabels(list(self.raw_data[self.categorical_clms[i]].dropna().unique()),
                                            rotation='vertical')
                plt.savefig('{}_{}.png'.format('Categorical', self.file_name))
                plt.gcf().clear()
                self._html_image_obj_creator('{}_{}.png'.format('Categorical', self.file_name), heading='')
            else:
                plt.gcf().clear()
                if 2 <= len(self.categorical_clms) <= 3:
                    row = 1
                    col = len(self.categorical_clms)
                    fig, axs = plt.subplots(row, col, figsize=(15, 5), facecolor='w', edgecolor='k')
                else:
                    row = int((len(self.categorical_clms)-(len(self.categorical_clms) % 3))/3)
                    col = 3
                    fig, axs = plt.subplots(row, col, figsize=(18, row*4), facecolor='w', edgecolor='k')
                axs = axs.ravel()
                for i in range(row*col):
                    temp.append(self.categorical_clms[i])
                    sns.countplot(self.categorical_clms[i], data=self.raw_data, ax=axs[i])
                    # axs[i].set_xlim(0, 5) # this is to reduce the width of the bar in bar plot
                    # x_pos = np.arange(len(self.raw_data[self.categorical_clms[i]].dropna().unique()))
                    if len(self.raw_data[self.categorical_clms[i]].unique()) > 20:
                        axs[i].set_xticklabels(['' for i in range(
                            len(self.raw_data[self.categorical_clms[i]].dropna().unique()))],
                                               rotation='vertical')
                plt.savefig('{}_{}.png'.format('Categorical', self.file_name))
                self._html_image_obj_creator('{}_{}.png'.format('Categorical', self.file_name), heading='')
                if len(temp) != len(self.categorical_clms):
                    plt.gcf().clear()
                    row = 1
                    col = len([i for i in self.categorical_clms if i not in temp])
                    left_out = [i for i in self.categorical_clms if i not in temp]
                    fig, axs = plt.subplots(row, len([i for i in self.categorical_clms if i not in temp]),
                                            figsize=(6*col-2, 4), facecolor='w', edgecolor='k')
                    for i in range(col):
                        if col == 1:
                            sns.countplot(left_out[i], data=self.raw_data, ax=axs)
                            if len(self.raw_data[left_out[i]].unique()) > 20:
                                axs.set_xticklabels(['' for i in range(
                                    len(self.raw_data[left_out[i]].dropna().unique()))],
                                                    rotation='vertical')
                        else:
                            sns.countplot(left_out[i], data=self.raw_data, ax=axs[i])
                            if len(self.raw_data[self.categorical_clms[i]].unique()) > 20:
                                axs[i].set_xticklabels(['' for i in range
                                                       (len(self.raw_data[left_out[i]].dropna().unique()))],
                                                       rotation='vertical')
                    plt.savefig('{}_v2_{}.png'.format('Categorical', self.file_name))
                    self._html_image_obj_creator('{}_v2_{}.png'.format('Categorical', self.file_name), heading='')

    def _numerical_dist_plot(self):
        self.HTML_content.append('<h1 style="font-size:1.5vw;">Distribution plot:</h1>')
        plt.gcf().clear()
        temp = []
        sns.set(color_codes=True)
        sns.set_palette(sns.color_palette("muted"))
        if len(self.numeric_clms) == 1:
            row = 1
            fig, axs = plt.subplots(row, figsize=(6, 7), facecolor='w', edgecolor='k')
            for i in range(row):
                sns.distplot(self.raw_data[self.numeric_clms[i]].dropna(), ax=axs, axlabel=False,
                             label=self.numeric_clms[i])
                axs.legend(["{}".format(self.numeric_clms[i])])
            plt.savefig('{}_{}.png'.format('Histogram', self.file_name))
            plt.gcf().clear()
            self._html_image_obj_creator('{}_{}.png'.format('Histogram', self.file_name), heading='')
        else:
            if 2 <= len(self.numeric_clms) <= 3:
                row = 1
                col = len(self.numeric_clms)
                fig, axs = plt.subplots(row, col, figsize=(15, 5), facecolor='w', edgecolor='k')
            else:
                row = int((len(self.numeric_clms)-(len(self.numeric_clms) % 3))/3)
                col = 3
                fig, axs = plt.subplots(row, col, figsize=(15, row*4), facecolor='w', edgecolor='k')

            axs = axs.ravel()
            for i in range(row*col):
                temp.append(self.numeric_clms[i])
                sns.distplot(self.raw_data[self.numeric_clms[i]].dropna(), ax=axs[i], axlabel=False,
                             label=self.numeric_clms[i])
                axs[i].legend(["{}".format(self.numeric_clms[i])])
            plt.savefig('{}_{}.png'.format('Histogram', self.file_name))
            plt.gcf().clear()
            self._html_image_obj_creator('{}_{}.png'.format('Histogram', self.file_name), heading='')
            if len(temp) != len(self.numeric_clms):
                plt.gcf().clear()
                row = 1
                col = len([i for i in self.numeric_clms if i not in temp])
                left_out = [i for i in self.numeric_clms if i not in temp]
                fig, axs = plt.subplots(row, len([i for i in self.numeric_clms if i not in temp]), figsize=(6*col-2, 4),
                                        facecolor='w', edgecolor='k')
                for i in range(col):
                    if col == 1:
                        sns.distplot(self.raw_data[left_out[i]].dropna(), ax=axs, axlabel=False, label=left_out[i])
                        axs.legend(["{}".format(left_out[i])])
                    else:
                        sns.distplot(self.raw_data[left_out[i]].dropna(), ax=axs[i], axlabel=False, label=left_out[i])
                        axs[i].legend(["{}".format(left_out[i])])
                plt.savefig('{}_v2_{}.png'.format('Histogram', self.file_name))
                self._html_image_obj_creator('{}_v2_{}.png'.format('Histogram', self.file_name), heading='')

    def _numerical_box_plot(self):
        self.HTML_content.append('<h1 style="font-size:1.5vw;">Box plot:</h1>')
        temp = []
        plt.gcf().clear()
        if len(self.numeric_clms) == 1:
            row = 1
            fig, axs = plt.subplots(row, figsize=(6, 6), facecolor='w', edgecolor='k')
            for i in range(row):
                sns.boxplot(x=self.raw_data[self.numeric_clms[i]].dropna(), showmeans=True, ax=axs, orient='v')
                axs.legend(["{}".format(self.numeric_clms[i])])
            plt.savefig('{}_{}.png'.format('Box', self.file_name))
            plt.gcf().clear()
            self._html_image_obj_creator('{}_{}.png'.format('Box', self.file_name), heading='')
        else:
            if 2 <= len(self.numeric_clms) <= 3:
                row = 1
                col = len(self.numeric_clms)
                fig, axs = plt.subplots(row, col, figsize=(15, 5), facecolor='w', edgecolor='k')
            else:
                row = int((len(self.numeric_clms)-(len(self.numeric_clms) % 3))/3)
                col = 3
                fig, axs = plt.subplots(row, col, figsize=(15, row*3), facecolor='w', edgecolor='k')
            axs = axs.ravel()
            for i in range(row * col):
                temp.append(self.numeric_clms[i])
                sns.boxplot(x=self.raw_data[self.numeric_clms[i]].dropna(), showmeans=True, ax=axs[i], orient='v')
                axs[i].legend(["{}".format(self.numeric_clms[i])])
            plt.savefig('{}_{}.png'.format('Box', self.file_name))
            plt.gcf().clear()
            self._html_image_obj_creator('{}_{}.png'.format('Box', self.file_name), heading='')
            if len(temp) != len(self.numeric_clms):
                plt.gcf().clear()
                row = 1
                col = len([i for i in self.numeric_clms if i not in temp])
                left_out = [i for i in self.numeric_clms if i not in temp]
                fig, axs = plt.subplots(row, len([i for i in self.numeric_clms if i not in temp]),
                                        figsize=(6*col-2, 4), facecolor='w', edgecolor='k')
                for i in range(col):
                    if col == 1:
                        sns.boxplot(x=self.raw_data[left_out[i]].dropna(), showmeans=True, ax=axs, orient='v')
                        axs.legend(["{}".format(left_out[i])])
                    else:
                        sns.boxplot(x=self.raw_data[left_out[i]].dropna(), showmeans=True, ax=axs[i], orient='v')
                        axs[i].legend(["{}".format(left_out[i])])
                plt.savefig('{}_v2_{}.png'.format('Box', self.file_name))
                self._html_image_obj_creator('{}_v2_{}.png'.format('Box', self.file_name), heading='')

    def _plot_corr(self):
        """
        Function plots a graphical correlation matrix for each pair of columns in the dataframe.

        Input:
            df: pandas DataFrame
            size: vertical and horizontal size of the plot
        """
        size = 7
        corr = self.raw_data[self.numeric_clms].corr()
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)

        image_file_name = '{}.png'.format('corr_plot')
        plt.savefig(image_file_name, bbox_inches='tight')
        self._html_image_obj_creator(image_file_name, heading='Correlation Plot')

    def _row_wise_missing_val(self):
        row_wise_miss_value_count = list(pd.isnull(self.raw_data).sum(axis=1))
        row_wise_miss_val_df = self.raw_data[pd.Series
                                             (row_wise_miss_value_count).apply(lambda val:val >=
                                                                               (math.ceil
                                                                                (0.50 * len(self.raw_data.columns))))]
        miss_per = len(row_wise_miss_val_df)/len(self.raw_data) * 100
        if len(row_wise_miss_val_df) > 0:
            self._html_df_obj_creator(row_wise_miss_val_df.sample(10),
                                      heading="""
                        Row-wise Missing Table: Total {}% data are missing for more than
                        50% of Variable (Showing sample of missing data)""".format(round(miss_per, 2)))

    def _del_image_file(self):
        # Deletes all the saved .png files created while generating the plots
        self.working_path_dir = os.getcwd()
        if (self.working_path_dir.split('/')[1] == 'home') & (self.working_path_dir.split('/')[2] == 'cdsw'):
            self.file_path = '{}{}'.format(self.working_path_dir, '/')
        else:
            self.file_path = '{}{}'.format(self.working_path_dir, '\\')
        direc = os.listdir(self.file_path)
        for item in direc:
            if item.endswith(".png"):
                os.remove(item)

    def _calc_skewness_kurtosis(self):
        plt.gcf().clear()
        # Adding Kurtosis warnings and plot
        self.kurtosis_df = pd.DataFrame(self.raw_data.kurtosis(), columns=['kurtosis'])
        self.kurtosis_df['kurtosis'] = self.kurtosis_df['kurtosis'].apply(lambda x: round(x, 2))
        kurt_warn = [val for val in self.kurtosis_df['kurtosis'] if val > 200]
        if len(kurt_warn) > 0:
            text_cursor = '<h1 style="font-size:1.2vw;">{}<br /></h1>'.format('Warnings : ')
            self.HTML_content.append(text_cursor)
            for i, row in self.kurtosis_df.iterrows():
                if row['kurtosis'] > 200:
                    message = '<li>{} have high kurtosis ( k = {})</li>'.format(i, row['kurtosis'])
                    self.HTML_content.append(message)

        # Adding Skewness warnings and plot
        self.skew_df = pd.DataFrame(self.raw_data.skew(), columns=['skewness'])
        self.skew_df['skewness'] = self.skew_df['skewness'].apply(lambda x: round(x, 2))
        skew_warn = [val for val in self.skew_df['skewness'] if val > 40]
        if len(skew_warn) > 0:
            for i, row in self.skew_df.iterrows():
                if row['skewness'] > 40:
                    message = '<li>{} have high skewness ( Y = {})</li>'.format(i, row['skewness'])
                    self.HTML_content.append(message)
        # Plotting kurtosis and skewness
        plt.gcf().clear()
        fig, axs = plt.subplots(1, 2, figsize=(10, 6), facecolor='w', edgecolor='k')
        axs = axs.ravel()
        self.kurtosis_df.sort_values(by=['kurtosis'], ascending=True).plot(kind='bar', ax=axs[0])

        self.skew_df.sort_values(by=['skewness'], ascending=True).plot(kind='bar', ax=axs[1])
        image_file_name = '{}.png'.format('kurtosis and Skew plot'.replace(' ', ''))
        plt.savefig(image_file_name, bbox_inches='tight')
        plt.gcf().clear()
        self._html_image_obj_creator(image_file_name, heading='Kurtosis and Skewness Plot')

    def _date_time_analysis(self):
        # ToDo : Create a seperate method for stand alone functionality
        for clm in self.date_time_clms:
            # Time period missing monthly
            date_time = pd.DataFrame({'count_col': list(self.raw_data[clm])}, index=self.raw_data[clm])
            del date_time.index.name
            # Reshaping the date column for plotting
            pv = pd.pivot_table(date_time, index=date_time.index.month, columns=date_time.index.year,
                                values='count_col', aggfunc='count')
            pv.index = pd.Series(pv.index).apply(lambda elem: int(elem))
            pv.columns = pd.Series(pv.columns).apply(lambda elem: int(elem))
            pv.index = pd.Series(pv.index).apply(lambda row: calendar.month_abbr[int(row)])
            plt.gcf().clear()
            fig, axs = plt.subplots(1, 2, figsize=(14, 4), facecolor='w', edgecolor='k')
            pv.plot(ax=axs[0])
            axs[0].legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
            # Time period missing daily
            pv = pd.pivot_table(date_time, index=date_time.index.day, columns=date_time.index.year,
                                values='count_col', aggfunc='count')
            pv.index = pd.Series(pv.index).apply(lambda elem: int(elem))
            pv.columns = pd.Series(pv.columns).apply(lambda elem: int(elem))
            pv.plot(ax=axs[1])
            axs[1].legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
            image_file_name = '{}.png'.format('Time plot daily {}'.format(clm).replace(' ', ''))
            plt.savefig(image_file_name, bbox_inches='tight')
            self._html_image_obj_creator(image_file_name,
                                         heading='''Exploring column {} on monthly and daily basis to see any
                                         missing time period (any discontinuity in curve
                                         shows data unavailability for that period)'''.format(clm))

            plt.gcf().clear()
            fig, axs = plt.subplots(1, 3, figsize=(15, 4), facecolor='w', edgecolor='k')
            axs = axs.ravel()
            # year wise count
            self.raw_data.groupby(self.raw_data[clm].dt.strftime('%Y'))[clm].count().plot(kind='line', ax=axs[0])
            # month wise count
            self.raw_data.groupby(self.raw_data[clm].dt.strftime('%B'))[clm].count().plot(kind='line', ax=axs[1])
            # day wise count
            self.raw_data.groupby(self.raw_data[clm].dt.strftime('%d'))[clm].count().plot(kind='line', ax=axs[2])
            image_file_name = '{}.png'.format('Time_Plot_Year_wise{}'.format(clm).replace(' ', ''))
            plt.savefig(image_file_name, bbox_inches='tight')
            self._html_image_obj_creator(image_file_name, heading='{}'.format(clm.title()))
            if self.target_variable:
                self._time_col_versus_target_analysis(clm)

    def _time_col_versus_target_analysis(self, clm):
        plt.gcf().clear()
        date_time = pd.DataFrame({'sum_col': list(self.raw_data[self.target_variable])}, index=self.raw_data[clm])
        del date_time.index.name
        pv = pd.pivot_table(date_time, index=date_time.index.month, columns=date_time.index.year,
                            values='sum_col', aggfunc='sum')
        pv.index = pd.Series(pv.index).apply(lambda elem: int(elem))
        pv.columns = pd.Series(pv.columns).apply(lambda elem: int(elem))
        pv.index = pd.Series(pv.index).apply(lambda row: calendar.month_abbr[int(row)])
        pv.plot(kind='bar', figsize=(12, 6))
        plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
        image_file_name = '{}.png'.format('target_vs_{}'.format(clm).replace(' ', ''))
        plt.savefig(image_file_name, bbox_inches='tight')
        self._html_image_obj_creator(image_file_name,
                                     heading='''Target variable - {} versus {}'''.format(self.target_variable.title(),
                                                                                         clm.title()))

    def _missing_val_vis_analysis(self):
        plt.gcf().clear()
        msno.matrix(self.raw_data, figsize=(12, 6), fontsize=12)
        image_file_name = '{}.png'.format('VisualMissingValAnalysis')
        plt.savefig(image_file_name, bbox_inches='tight')
        plt.gcf().clear()
        self._html_image_obj_creator(image_file_name,
                                     heading='''Missing Value Visual plot, Right hand side shows
                                     the row wise missing info''')

    def profile(dataframe=None, file_name='analysis_report', target_variable=None, date_time_var=None,
                col_to_be_excluded=None):
        col_to_be_excluded = col_to_be_excluded
        target_variable = target_variable
        date_time_var = date_time_var
        # Data Profiling
        if dataframe is None:
            sys.exit("Please input pandas data frame")
        eda = ExploratoraryDataAnalysis()
        eda.data_profiling(dataframe, file_name, target_variable, date_time_var, col_to_be_excluded)
        # creating head
        eda._create_report_head()
        # create data summary
        eda._create_data_summary()
        # calculating skewness and kurtosis
        eda._calc_skewness_kurtosis()
        # Creating missing value df
        eda._miss_value_df_creator()
        # Missing data visual analysis
        # Creating distinct value df
        eda._distinct_value_df_creator()
        # merge miss and distinct df
        eda._miss_distinct_df_creator()
        # plot miss distinct plot
        eda._missing_distinct_plot()
        eda._missing_val_vis_analysis()
        # categorical attribute count plot
        eda._categorical_value_count_plot()
        # numerical attribute distribution plot
        eda._numerical_dist_plot()
        # numerical attribute box plot
        eda._numerical_box_plot()
        # creating row wise miss value df
        eda._row_wise_missing_val()
        # plot correlation matrix
        eda._plot_corr()
        # Time series analysis
        eda._date_time_analysis()
        # writing the HTML
        eda._html_writer()
        print("Report Created")
        # Clean the .png file
        eda._del_image_file()
