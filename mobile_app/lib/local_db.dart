import 'dart:convert';
import 'dart:math' as math;
import 'package:http/http.dart' as http;
import 'package:sqflite/sqflite.dart';
import 'package:uuid/uuid.dart';

class PendingPotholeRecord {
  final String eventUuid;
  final double latitude;
  final double longitude;
  final double speedKmh;
  final double confidence;
  final double severityScore;
  final String lanePosition;
  final String capturedAt;
  final String? imageBase64;

  const PendingPotholeRecord({
    required this.eventUuid,
    required this.latitude,
    required this.longitude,
    required this.speedKmh,
    required this.confidence,
    required this.severityScore,
    required this.lanePosition,
    required this.capturedAt,
    this.imageBase64,
  });

  Map<String, dynamic> toMap() => {
        'event_uuid': eventUuid,
        'latitude': latitude,
        'longitude': longitude,
        'speed_kmh': speedKmh,
        'confidence': confidence,
        'severity_score': severityScore,
        'lane_position': lanePosition,
        'captured_at': capturedAt,
        'image_base64': imageBase64,
      };

  factory PendingPotholeRecord.fromMap(Map<String, dynamic> map) {
    return PendingPotholeRecord(
      eventUuid: map['event_uuid'] as String,
      latitude: (map['latitude'] as num).toDouble(),
      longitude: (map['longitude'] as num).toDouble(),
      speedKmh: (map['speed_kmh'] as num).toDouble(),
      confidence: (map['confidence'] as num).toDouble(),
      severityScore: (map['severity_score'] as num).toDouble(),
      lanePosition: map['lane_position'] as String,
      capturedAt: map['captured_at'] as String,
      imageBase64: map['image_base64'] as String?,
    );
  }
}

class LocalHazardQueue {
  final Database db;
  static const String tableName = 'pending_potholes';
  static const double dedupRadiusMeters = 3.0;
  static const int maxBatchSize = 50;
  static const _uuid = Uuid();

  LocalHazardQueue(this.db);

  static Future<LocalHazardQueue> open(String dbPath, {DatabaseFactory? factory}) async {
    final dbFactory = factory ?? databaseFactory;
    final database = await dbFactory.openDatabase(
      dbPath,
      options: OpenDatabaseOptions(
        version: 1,
        onCreate: (db, version) async {
          await db.execute('''
            CREATE TABLE $tableName (
              event_uuid TEXT PRIMARY KEY,
              latitude REAL NOT NULL,
              longitude REAL NOT NULL,
              speed_kmh REAL NOT NULL,
              confidence REAL NOT NULL,
              severity_score REAL NOT NULL,
              lane_position TEXT NOT NULL,
              captured_at TEXT NOT NULL,
              image_base64 TEXT
            )
          ''');
        },
      ),
    );
    return LocalHazardQueue(database);
  }

  static double haversineMeters(double lat1, double lon1, double lat2, double lon2) {
    const earthRadiusM = 6371000.0;
    final dLat = (lat2 - lat1) * math.pi / 180.0;
    final dLon = (lon2 - lon1) * math.pi / 180.0;
    final a = math.sin(dLat / 2) * math.sin(dLat / 2) +
        math.cos(lat1 * math.pi / 180.0) *
            math.cos(lat2 * math.pi / 180.0) *
            math.sin(dLon / 2) *
            math.sin(dLon / 2);
    final c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a));
    return earthRadiusM * c;
  }

  Future<bool> enqueueIfNotDuplicate({
    required double latitude,
    required double longitude,
    required double speedKmh,
    required double confidence,
    required double severityScore,
    required String lanePosition,
    String? imageBase64,
    String? customUuid,
  }) async {
    final existing = await getAllPending();
    final isDuplicate = existing.any((item) =>
        haversineMeters(latitude, longitude, item.latitude, item.longitude) <
        dedupRadiusMeters);

    if (isDuplicate) {
      return false;
    }

    final record = PendingPotholeRecord(
      eventUuid: customUuid ?? _uuid.v4(),
      latitude: latitude,
      longitude: longitude,
      speedKmh: speedKmh,
      confidence: confidence,
      severityScore: severityScore,
      lanePosition: lanePosition,
      capturedAt: DateTime.now().toUtc().toIso8601String(),
      imageBase64: imageBase64,
    );

    await db.insert(
      tableName,
      record.toMap(),
      conflictAlgorithm: ConflictAlgorithm.ignore,
    );
    return true;
  }

  Future<List<PendingPotholeRecord>> getAllPending() async {
    final rows = await db.query(tableName, orderBy: 'captured_at ASC');
    return rows.map(PendingPotholeRecord.fromMap).toList();
  }

  Future<int> flushBatchToBackend({
    required String apiBaseUrl,
    required String deviceId,
    http.Client? httpClient,
  }) async {
    final client = httpClient ?? http.Client();
    final pending = await getAllPending();
    if (pending.isEmpty) return 0;

    final batch = pending.take(maxBatchSize).toList();
    final payload = {
      'device_id': deviceId,
      'detections': batch.map((e) => e.toMap()).toList(),
    };

    try {
      final response = await client.post(
        Uri.parse('$apiBaseUrl/api/v1/telemetry/sync'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;
        final syncedUuids = (decoded['synced_uuids'] as List<dynamic>)
            .map((e) => e.toString())
            .toList();

        if (syncedUuids.isNotEmpty) {
          final placeholders = List.filled(syncedUuids.length, '?').join(',');
          await db.delete(
            tableName,
            where: 'event_uuid IN ($placeholders)',
            whereArgs: syncedUuids,
          );
        }
        return syncedUuids.length;
      }
      return 0;
    } catch (_) {
      return 0;
    }
  }
}
