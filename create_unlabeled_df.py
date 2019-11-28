# import os
# import pandas as pd
# import glob

# #create list of file paths
# paths = []
# for filepath in glob.iglob('/home/mo/pardis/unlabelled_comments/*'):
#     paths.append(filepath)
# print(paths)



# for file_path in paths:
# 	file = open(file_path, 'r')
# 	ul_data['comments'] = ul_data['exact_comments'] = file.readlines()
# print(ul_data)
# # path = './unlabelled_comments'
# # path2 = '/home/mo/pardis/unlabelled_comments/depression_comments.csv'
# # # for comment_file in os.listdir(path):
# # #    # print(filename)
# # #     with open(comment_file, 'r') as file:
# # #     	comments = file.readlines()
# # #     	print(comments.size)

# # ul_data = pd.DataFrame(columns=['comments', 'exact_comments', 'polarity', 'confidence_score'])
# # file = open(path2, 'r')
# # ul_data['comments'] = ul_data['exact_comments'] = file.readlines()
# # print(ul_data)



import pandas as pd
import glob

def create_df_for_unlabeled_data():
	ul_data = pd.DataFrame(columns=['comments', 'exact_comments', 'polarity', 'confidence_score'])
	path = r'/home/mo/pardis/unlabelled_comments' # use your path
	all_files = glob.glob(path + "/*.csv")

	all_comments = []

	for filename in all_files:
	    df = pd.read_csv(filename, index_col=None, header=None)
	    # li.append(df)
	    file = open(filename, 'r')
	    all_comments.append(file.readlines())
	# print(all_comments)

	comment = []
	for j in range(len(all_comments)):
		for element in all_comments[j]:
			stripped_comment = str(element).split('\\n')
			# print(stripped_comment)
			stripped_n_comment= str(stripped_comment).replace('\\n','')
			print(stripped_n_comment)
			comment.append(stripped_n_comment)
		print(len(comment))
		# frame = pd.concat(comment, axis=0, ignore_index=True, sort = False)
		# print(frame)
	ul_data['comments'] = ul_data['exact_comments'] = comment
	return ul_data
	# print(ul_data)