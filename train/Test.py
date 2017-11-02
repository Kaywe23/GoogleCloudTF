


csv_file='train_converted_vermischt.csv'

gcs_file = pd.read_csv(tf.gfile.Open(csv_file, 'rb'), encoding='latin-1', buffer_lines=20000, skipinitialspace=True)
gcs_file.to_csv()
# csv_reader=gcs_file.read()
# with io.open(csv_file1, buffering=20000, encoding='latin-1') as f:
# df_train = pd.read_csv(tf.gfile.Open("./train.csv"),skipinitialspace=True)