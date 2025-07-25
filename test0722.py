from ApolloDataProcessor import ApolloMSeedProcessor


processor = ApolloMSeedProcessor("/home/tu/data/noise", "/home/tu/code/OUTPUT_FILES")
mseed_mat = "mat_mseed"
df1, df2, df3 = processor.process_mseed_files(mseed_mat)

print(df1.head())
print(df1.info())