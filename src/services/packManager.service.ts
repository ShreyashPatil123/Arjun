import { invoke } from '@tauri-apps/api/core';
import type { PackageCertification, RuntimeProfile } from '../types/recommendation';

export async function getPackageCertification(modelId: string): Promise<PackageCertification | null> {
  try {
    return await invoke<PackageCertification | null>('get_package_certification', { modelId });
  } catch (err) {
    console.error('Failed to get package certification:', err);
    return null;
  }
}

export async function getAllPackageCertifications(): Promise<PackageCertification[]> {
  try {
    return await invoke<PackageCertification[]>('get_all_package_certifications');
  } catch (err) {
    console.error('Failed to get all package certifications:', err);
    return [];
  }
}

export async function getRuntimeProfile(profileId: string): Promise<RuntimeProfile | null> {
  try {
    return await invoke<RuntimeProfile | null>('get_runtime_profile', { profileId });
  } catch (err) {
    console.error('Failed to get runtime profile:', err);
    return null;
  }
}

export async function reloadCertificationPacks(): Promise<void> {
  try {
    await invoke<void>('reload_certification_packs');
  } catch (err) {
    console.error('Failed to reload certification packs:', err);
  }
}
