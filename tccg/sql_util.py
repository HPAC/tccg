# Copyright (C) 2016 Paul Springer (springer@aices.rwth-aachen.de) - All Rights Reserved
import sqlite3
import socket
import traceback

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

def getLastPrimaryKey(cursor):
    command = "SELECT last_insert_rowid()"
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], ENDC
        traceback.print_exc()   
        exit(-1)
    result = cursor.fetchall() 
    return result[0][0]

def getFastestLoop(cursor, measurementID):
    command = "select flops from loop where measurement_id = %d"%(measurementID)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        exit(-1)

    maxFlops = -1
    result = cursor.fetchall() 
    if( len(result) > 0 ):
        for r in result:
            if( r[0] > maxFlops ):
                maxFlops = r[0]
        return (maxFlops)
    else:
        return (maxFlops)

def getFastestTTGT(cursor, measurementID):
    command = "select flops from ttgt where measurement_id = %d"%(measurementID)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        exit(-1)

    maxFlops = -1
    result = cursor.fetchall() 
    if( len(result) > 0 ):
        for r in result:
            if( r[0] > maxFlops ):
                maxFlops = r[0]
        return (maxFlops)
    else:
        return (maxFlops)



def getFastestGETT(cursor, measurementID):
    command = "select flops, variant, mc, nc,kc,mc1, nc1, mr,nr,indices from gett where measurement_id = %d"%(measurementID)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        exit(-1)

    maxFlops = -1
    variant = ""
    mc = ""
    nc = ""
    kc = ""
    mc1 = ""
    nc1 = ""
    mr = ""
    nr = ""
    indices = ""
    result = cursor.fetchall() 
    if( len(result) > 0 ):
        for r in result:
            if( r[0] > maxFlops ):
                maxFlops = r[0]
                variant = r[1]
                mc =  r[2]
                nc = r[3]
                kc =  r[4]
                mc1 = r[5]
                nc1 = r[6]
                mr = r[7]
                nr =  r[8]
                indices =  r[9]
        return (maxFlops, variant, mc, nc, kc , mc1 , nc1 , mr , nr, indices)
    else:
        return (maxFlops, variant, mc, nc, kc , mc1 , nc1 , mr , nr, indices)


def insertIntoGETT(cursor, variant,
                  flops, 
                  estimatedFlops, 
                  mc ,
                  nc,
                  kc ,
                  mc1,
                  nc1,
                  mr,
                  nr,
                  indices,
                  measurement_id):
    command = """INSERT INTO gett(
                  variant,
                  flops, 
                  estimatedFlops, 
                  mc ,
                  nc,
                  kc ,
                  mc1,
                  nc1,
                  mr,
                  nr,
                  indices,
                  measurement_id)
                VALUES ( 
                '%s', 
                %f,
                %f,
                %d, 
                %d, 
                %d, 
                %d, 
                %d, 
                %d, 
                %d,
                '%s',
                %d 
                );"""%( variant,
                  flops, 
                  estimatedFlops, 
                  mc ,
                  nc,
                  kc ,
                  mc1,
                  nc1,
                  mr,
                  nr,
                  indices,
                  measurement_id)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)
    primaryKey = getLastPrimaryKey(cursor)
    return primaryKey

def insertIntoGEMM(cursor, measurement_id, flops):
    command = """INSERT INTO gemm(
                          measurement_id, flops)
                VALUES ( 
                %d, 
                %f
                );"""%( measurement_id, flops)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)
    primaryKey = getLastPrimaryKey(cursor)
    return primaryKey


def insertIntoTTGT(cursor, measurement_id, flops):
    command = """INSERT INTO ttgt(
                          measurement_id, flops)
                VALUES ( 
                %d, 
                %f
                );"""%( measurement_id, flops)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)
    primaryKey = getLastPrimaryKey(cursor)
    return primaryKey

def insertIntoLoop(cursor, measurement_id, flops):
    command = """INSERT INTO loop(
                          measurement_id, flops)
                VALUES ( 
                %d, 
                %f
                );"""%( measurement_id, flops)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)
    primaryKey = getLastPrimaryKey(cursor)
    return primaryKey


def getMeasurementId(cursor,
                          tensorA,
                          tensorB,
                          tensorC,
                          size,
                          architecture,
                          numThreads,
                          floatTypeA,
                          floatTypeB,
                          floatTypeC,
                          beta):
    if( beta != 0 ): # don't distinguis different non-zero beta values
        beta = 1
    command = "select measurement_id from measurements where tensorA = '%s' and tensorB = '%s' and tensorC = '%s' and size = '%s' and architecture = '%s' and numThreads = %d and floatTypeA = '%s' and floatTypeB = '%s' and floatTypeC = '%s' and beta = %d"%(tensorA,
                          tensorB,
                          tensorC,
                          size,
                          architecture,
                          numThreads,
                          floatTypeA,
                          floatTypeB,
                          floatTypeC,
                          beta)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "ERROR (sql):", e.args[0], ENDC
        exit(-1)

    result = cursor.fetchall() 
    if( len(result) > 0 ):
        return result[0][0]
    else:
        return -1

def insertIntoMeasurements(cursor,
                          tensorA,
                          tensorB,
                          tensorC,
                          size,
                          architecture,
                          numThreads,
                          compiler_version,
                          floatTypeA,
                          floatTypeB,
                          floatTypeC,
                          referenceFlops, gemmFlops, beta):
    hostname = socket.gethostname()
    if( beta != 0 ): # don't distinguis different non-zero beta values
        beta = 1
    
    command = """INSERT INTO measurements (
                          tensorA,
                          tensorB,
                          tensorC,
                          size,
                          architecture,
                          host,
                          numThreads,
                          compiler_version,
                          floatTypeA,
                          floatTypeB,
                          floatTypeC,
                          referenceFlops,
                          gemmFlops,
                          beta )
                VALUES ( 
                '%s', 
                '%s',
                '%s',
                '%s',
                '%s',
                '%s',
                %d,
                '%s',
                '%s',
                '%s',
                '%s',
                %f,
                %f,
                %f
                );"""%(   tensorA,
                          tensorB,
                          tensorC,
                          size,
                          architecture,
                          hostname,
                          numThreads,
                          compiler_version,
                          floatTypeA,
                          floatTypeB,
                          floatTypeC,
                          referenceFlops, gemmFlops, beta)
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)
    primaryKey = getLastPrimaryKey(cursor)
    return primaryKey


def createTables(cursor):
    command = """
    CREATE TABLE IF NOT EXISTS 'loop' (
      'loop_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'flops' FLOAT NULL, 
      'measurement_id' INTEGER NULL ,
      FOREIGN KEY ('measurement_id') REFERENCES 'measurements' ('measurement_id')
      )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)

    command = """
    CREATE TABLE IF NOT EXISTS 'ttgt' (
      'ttgt_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'flops' FLOAT NULL, 
      'measurement_id' INTEGER NULL ,
      FOREIGN KEY ('measurement_id') REFERENCES 'measurements' ('measurement_id')
      )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)

    command = """
    CREATE TABLE IF NOT EXISTS 'gemm' (
      'gemm_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'flops' FLOAT NULL, 
      'measurement_id' INTEGER NULL ,
      FOREIGN KEY ('measurement_id') REFERENCES 'measurements' ('measurement_id')
      )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)


    command = """
    CREATE TABLE IF NOT EXISTS 'gett' (
      'gett_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'variant' VARCHAR(60) NULL ,
      'flops' FLOAT NULL, 
      'estimatedFlops' FLOAT NULL, 
      'mc' INTEGER NULL ,
      'nc' INTEGER NULL ,
      'kc' INTEGER NULL ,
      'mc1' INTEGER NULL ,
      'nc1' INTEGER NULL ,
      'mr' INTEGER NULL ,
      'nr' INTEGER NULL ,
      'indices' VARCHAR(100) NULL ,
      'measurement_id' INTEGER NULL ,
      FOREIGN KEY ('measurement_id') REFERENCES 'measurements' ('measurement_id')
      )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command,ENDC
        traceback.print_exc()   
        exit(-1)


    command = """
    CREATE TABLE IF NOT EXISTS 'measurements' (
      'measurement_id' INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT ,
      'tensorA' VARCHAR(45) NULL ,
      'tensorB' VARCHAR(45) NULL ,
      'tensorC' VARCHAR(45) NULL ,
      'size' VARCHAR(45) NULL ,
      'architecture' VARCHAR(45) NULL ,
      'host' VARCHAR(60) NULL ,
      'numThreads' INTEGER NULL ,
      'compiler_version' VARCHAR(60) NULL ,
      'floatTypeA' VARCHAR(45) NULL ,
      'floatTypeB' VARCHAR(45) NULL ,
      'floatTypeC' VARCHAR(45) NULL ,
      'referenceFlops' FLOAT NULL,
      'gemmFlops' FLOAT NULL,
      'beta' FLOAT NULL
      )
    """
    try:
        cursor.execute(command)
    except sqlite3.Error as e:
        print FAIL + "[TCC] ERROR (sql):", e.args[0], command, ENDC
        traceback.print_exc()   
        exit(-1)


