import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:mobile_app/local_db.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  sqfliteFfiInit();

  test('T-2.2.1: enqueueIfNotDuplicate persists new pothole and blocks <3m duplicate', () async {
    final queue = await LocalHazardQueue.open(
      inMemoryDatabasePath,
      factory: databaseFactoryFfi,
    );

    final firstInserted = await queue.enqueueIfNotDuplicate(
      latitude: 9.931200,
      longitude: 76.267300,
      speedKmh: 42.0,
      confidence: 0.88,
      severityScore: 6.5,
      lanePosition: 'CENTER',
    );
    final duplicateNearby = await queue.enqueueIfNotDuplicate(
      latitude: 9.931210,
      longitude: 76.267305,
      speedKmh: 41.0,
      confidence: 0.85,
      severityScore: 6.2,
      lanePosition: 'CENTER',
    );

    expect(firstInserted, isTrue);
    expect(duplicateNearby, isFalse);
    expect((await queue.getAllPending()).length, 1);
    await queue.db.close();
  });

  test('T-2.2.2: flushBatchToBackend slices at most 50 records per POST request', () async {
    final queue = await LocalHazardQueue.open(
      inMemoryDatabasePath,
      factory: databaseFactoryFfi,
    );

    for (var i = 0; i < 65; i++) {
      await queue.enqueueIfNotDuplicate(
        latitude: 9.9312 + (i * 0.001),
        longitude: 76.2673,
        speedKmh: 35.0,
        confidence: 0.90,
        severityScore: 5.0,
        lanePosition: 'LEFT',
      );
    }

    int postedCount = 0;
    final mockClient = MockClient((request) async {
      final body = jsonDecode(request.body) as Map<String, dynamic>;
      final detections = body['detections'] as List<dynamic>;
      postedCount = detections.length;
      final uuids = detections.map((d) => d['event_uuid']).toList();
      return http.Response(
        jsonEncode({'synced_uuids': uuids, 'total_processed': uuids.length}),
        200,
      );
    });

    final synced = await queue.flushBatchToBackend(
      apiBaseUrl: 'http://localhost:8000',
      deviceId: 'bike-01',
      httpClient: mockClient,
    );

    expect(postedCount, 50);
    expect(synced, 50);
    expect((await queue.getAllPending()).length, 15);
    await queue.db.close();
  });

  test('T-2.2.3: flushBatchToBackend retains unsynced records on network failure', () async {
    final queue = await LocalHazardQueue.open(
      inMemoryDatabasePath,
      factory: databaseFactoryFfi,
    );

    await queue.enqueueIfNotDuplicate(
      latitude: 9.9312,
      longitude: 76.2673,
      speedKmh: 40.0,
      confidence: 0.91,
      severityScore: 7.0,
      lanePosition: 'RIGHT',
    );

    final failingClient = MockClient((_) async => http.Response('Server Error', 500));
    final synced = await queue.flushBatchToBackend(
      apiBaseUrl: 'http://localhost:8000',
      deviceId: 'bike-01',
      httpClient: failingClient,
    );

    expect(synced, 0);
    expect((await queue.getAllPending()).length, 1);
    await queue.db.close();
  });
}
