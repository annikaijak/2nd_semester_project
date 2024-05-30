Data contains data from 9 sensors, 3 sensors installed at an electric charger spot, 3 sensors installed in a normal spot and 3 sensors installed in a forbidden parking area.

The columns are the following:
•	Time: time of measurement
•	Battery: battery voltage at packet transmission (the nominal voltage of the battery package is 3V, and the capacity is 6AH)
•	X: magnetic field reading in the x direction
•	Y: magnetic field reading in the y direction
•	Z: magnetic field reading in the z direction
•	For the 3 normal spot sensors(NORM1-2-3.csv):
    o	0_radar: signal reflection strength in the 20 – 25 cm
    o	1_radar: signal reflection strength in the 25 – 30 cm
    o	2_radar: signal reflection strength in the 30 –35 cm
    o	3_radar: signal reflection strength in the 35 – 40 cm
    o	4_radar: signal reflection strength in the 40 – 45 cm
    o	5_radar: signal reflection strength in the 45 – 50 cm
    o	6_radar: signal reflection strength in the 50 – 55 cm
    o	7_radar: signal reflection strength in the 55 – 60 cm
•	or the 6 remaining sensors (this radar scan length is the latest and will be used in all future sensors):
    o	0_radar: signal reflection strength in the 20 – 27.5 cm
    o	1_radar: signal reflection strength in the 27.5 – 35 cm
    o	2_radar: signal reflection strength in the 35 –42.5 cm
    o	3_radar: signal reflection strength in the 42.5 – 50 cm
    o	4_radar: signal reflection strength in the 50 – 57.5 cm
    o	5_radar: signal reflection strength in the 57.5 – 65 cm
    o	6_radar: signal reflection strength in the 65 – 72.5 cm
    o	7_radar: signal reflection strength in the 72.5 – 80 cm
•	Package_type:
    o	PackageType.CHANGE = a big change in magnetic field is detected
    o	PackageType.HEART_BEAT = regular heartbeat if no major changes in magnetic field are detected
•	f_cnt: number of packages transmitted since last network registration
•	dr: data rate parameter in LoRaWAN. It ranges between 1 and 5 where 1 is the slowest transmission data rate and 5 is the highest. This datarate is scaled by the network server depending on the signal quality of the past packages send.
•	SNR: signal to noise ratio – the higher value, the better the signal quality
•	RSSI: signal strength – the higher value, the better the signal quality


•	For information about the weather data visit https://open-meteo.com/en/docs
