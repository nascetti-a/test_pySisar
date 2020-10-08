import gdal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os, sys, getopt
import scipy.stats


# validazione_DSM.py Lidar_Roma2x2.tif CUT_DSM_tot_2m_goturk_Roma_ns_1_nd_64_MD_-16_SAD_5.TIF 1 64 -16 5 roma
# run validazione_DSM.py Lidar_corretto_Trento.tif AgisoftDEM1_WGS84_UTM32N_EPSG32632_orthoEGM96_1x1_ritagliato.tif Gokturk
# run validazione_DSM.py Lidar_corretto_Trento.tif DSM_Pleiades_tripletta_2020_01_09_1x1_ritagliato.tif Pleiades
# run validazione_DSM.py Lidar_corretto_Trento.tif DSM_Pleiades_pair1_2020_01_10_1x1_ritagliato.tif Pleiades
# run validazione_DSM.py Lidar_corretto_Trento.tif DSM_Pleiades_pair2_2020_01_10_1x1_ritagliato.tif Pleiades
# run validazione_DSM.py Lidar_corretto_Trento.tif DSM_Pleiades_pair3_2020_01_10_1x1_ritagliato.tif Pleiades
# run validazione_DSM.py Lidar_corretto_Trento.tif DSM_Pleiades_tripletta_merged_2020_01_10_1x1_ritagliato.tif Pleiades


# plt.close('all')
# path = os.path.dirname(os.path.abspath(__file__)) + '/'
# print
# path + '\n'
# # If not already present, create the directory to store the results
#



def LE(data, LE_value):
    # http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/wkflw-elevation-part1.htm
    Threshold_zmin = (100 - LE_value) / 2.0
    Threshold_zmax = Threshold_zmin + LE_value
    percentile_min = np.percentile(data, Threshold_zmin)
    percentile_max = np.percentile(data, Threshold_zmax)
    LE = (percentile_max - percentile_min) / 2.0
    return LE


def NMAD(data):
    median = np.median(data)
    MAD = np.absolute(data - median)
    NMAD = 1.4826 * np.median(MAD)
    return NMAD


def main():

    root = '/Volumes/Samsung_T5/DEM_Extraction_pyDATE/Trento/'
    inputfile2 = root + 'pyDATE'
    inputfile1 = root + 'Lidar_Ref'

    results_directory = 'ValidationResults_pyDate'

    path = root

    if not os.path.exists(path + '/' + results_directory):
        os.makedirs(path + '/' + results_directory)
    # ns = ''
    # nd = ''
    # MD = ''
    # SAD = ''
    # pair = ''
    # try:
    #     opts, args = getopt.getopt(argv, "h:o:", ["help=", "ofile="])
    #     print
    #     "OPTIONS", opts
    #     print
    #     "ARGUMENTS", args
    # except getopt.GetoptError:
    #     print
    #     'test.py <inputfile_lidar> <inputfile_DSM> <steps number> <ndisparities> <minimumDisp> <SADWindowSize> <pair>'
    #     sys.exit(2)
    #
    # if len(args) != 3:
    #     print
    #     'Inserire 3 argomenti, servono due DSM per la validazione, il nome del satellite'
    #     sys.exit(3)
    #
    # inputfile1 = args[0]
    # inputfile2 = args[1]

    satellite = "pleiades"  # args[2]

    # ns = args[2]
    # nd = args[3]
    # MD = args[4]
    # SAD = args[5]
    # pair = args[6]

    # for opt, arg in opts:
    #     if opt == '-h':
    #         print
    #         'test.py <inputfile_lidar> <inputfile_DSM> <steps number> <ndisparities> <minimumDisp> <SADWindowSize> <pair>'
    #         sys.exit()
    print
    'REF is ', inputfile1
    print
    'DEM is ', inputfile2

    lidar = gdal.Open(inputfile1)
    DSM = gdal.Open(inputfile2)

    DSM_data = DSM.ReadAsArray()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # DSM_data ha una riga e una colonna in più di reference_data
    print
    'DEM', DSM_data.shape
    # print 'REF', reference_data.shape
    #DSM_data = DSM_data[:-1, :-1]
    # DSM_data = DSM_data[:-1,:]

    DSM_data = DSM_data - 49.0

    # trasformo -9999 in nan
    indici = np.where(DSM_data < -500)
    ix = indici[1]  # x index
    iy = indici[0]  # y index
    DSM_data[iy, ix] = float('NaN')

    reference_data = lidar.ReadAsArray()
    # trasformo -9999 in nan
    indici = np.where(reference_data < -500)
    ix = indici[1]  # x index
    iy = indici[0]  # y index
    reference_data[iy, ix] = float('NaN')

    plt.figure('Agisoft DSM')
    plt.imshow(DSM_data, cmap=cm.gray, interpolation='None')
    cb = plt.colorbar()
    # cb.set_clim(0, 150)
    plt.savefig(path + results_directory + '/Agisoft_DSM_' + satellite + '.png', bbox_inches='tight')
    # pdb.set_trace()

    plt.figure('Reference Lidar')
    plt.imshow(reference_data, cmap=cm.gray, interpolation='None')
    cb = plt.colorbar()
    # cb.set_clim(0, 150)
    plt.savefig(path + results_directory + '/Reference_DSM.png', bbox_inches='tight')

    print
    'DEM', DSM_data.shape
    print
    'REF', reference_data.shape
    print
    '\n'
    # pdb.set_trace()
    ElevationDifference = reference_data - DSM_data
    print
    'not nan', np.count_nonzero(~np.isnan(ElevationDifference)), ElevationDifference.shape[0] * \
    ElevationDifference.shape[1]
    size = ElevationDifference.shape
    tot_points = len(ElevationDifference) * len(ElevationDifference[0])

    Dz = 50  # 100

    Partial_good_values = ElevationDifference[np.where(ElevationDifference < Dz)]

    Final_good_values = Partial_good_values[np.where(Partial_good_values > -Dz)]

    stats = np.zeros((11))
    stats_labels = np.zeros(11)
    stats_labels = np.array(
        ['mean', 'std', 'RMSE', 'median', 'NMAD', 'le68', 'le90', 'max', 'min', 'skeweness', 'kurtosis'])
    skewness = scipy.stats.skew(Final_good_values)  # indice di simmetria
    kurtosis = scipy.stats.kurtosis(Final_good_values)  # indice di allontanamento dalla distribuzione normale
    mean = np.mean(Final_good_values)
    std = np.std(Final_good_values)
    RMSE = np.sqrt(mean ** 2 + std ** 2)
    median = np.median(Final_good_values)
    nmad = NMAD(Final_good_values)
    le68 = LE(Final_good_values, 68)
    le90 = LE(Final_good_values, 90)
    max = np.nanmax(Final_good_values)
    min = np.nanmin(Final_good_values)

    stats[0] = mean
    stats[1] = std
    stats[2] = RMSE
    stats[3] = median
    stats[4] = nmad
    stats[5] = le68
    stats[6] = le90
    stats[7] = max
    stats[8] = min
    stats[9] = skewness
    stats[10] = kurtosis

    print('ElevationDifference           RMSE', format(RMSE, '.4f'))
    print('ElevationDifference           mean', format(mean, '.4f'))
    print('ElevationDifference            std', format(std, '.4f'))
    print('ElevationDifference         median', format(median, '.4f'))
    print('ElevationDifference           NMAD', format(nmad, '.4f'))
    print('ElevationDifference           LE68', format(le68, '.4f'))
    print('ElevationDifference           LE90', format(le90, '.4f'))
    print('ElevationDifference            max', format(max, '.4f'))
    print('ElevationDifference            min', format(min, '.4f'))
    print('ElevationDifference skewness coeff', format(skewness, '.3f')),
    if skewness < 0:
        print
        ': gobba a destra della media'
    elif skewness > 0:
        print
        ': gobba a sinistra della media'
    else:
        print
        ': campione simmetrico'
    print
    'ElevationDifference kurtosis coeff', format(kurtosis, '.3f'),
    if kurtosis < 0:
        print
        ': distribuzione platicurtica, ovvero maggiormente "piatta" rispetto ad una normale'
    elif kurtosis > 0:
        print
        ': distribuzione leptocurtica, ovvero maggiormente "appuntita" rispetto ad una normale'
    else:
        print
        ': distribuzione normocurtica (o mesocurtica), ovvero "piatta" come una normale'

    ###### STAMPA RISULTATI SU FILE ######

    file_stats = open(path + results_directory + "/global_statistic_parameters_" + satellite + ".txt", "w")

    for index in range(stats.shape[0]):
        file_stats.write(stats_labels[index] + '\t')

    file_stats.write('\n')

    file_stats.write('mean\t')

    for index in range(stats.shape[0]):
        file_stats.write(format(np.mean(stats[index]), '.4f') + '\t')

    file_stats.write('\nstd\t')
    for index in range(stats.shape[0]):
        file_stats.write(format(np.std(stats[index]), '.4f') + '\t')
    file_stats.close()

    plt.figure('Differences')
    plt.title('Reference - Agisoft DSM')
    plt.imshow(ElevationDifference, cmap=cm.jet, interpolation='None', vmin = -2*std, vmax = 2*std)
    # plt.imshow(ElevationDifference, interpolation = 'bicubic',  cmap = 'RdBu')
    # plt.colorbar(spacing ='uniform', extend ='both')
    plt.colorbar()
    # plt.clim(-std, std)
    # plt.clim(-50, 50)
    # plt.imsave('Elevation_difference', ElevationDifference, vmin = -std, vmax = std, cmap = 'RdBu', format = 'TIF')
    plt.savefig(path + results_directory + '/ElevationDifference_' + satellite + '.png', bbox_inches='tight')
    plt.grid(True)

    # pdb.set_trace()
    x = ElevationDifference.ravel().copy()
    xmask = x[np.logical_not(np.isnan(x))]
    plt.figure('Histo')
    n, bins, patches = plt.hist(xmask, bins='auto', density=False, facecolor='g', alpha=0.75)
    plt.savefig(path + results_directory + '/histo_' + satellite + '.png')
    plt.grid()
    # plt.grid(axis='y', alpha=0.75)
    '''plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    '''

    plt.show()


if __name__ == "__main__":
    main()