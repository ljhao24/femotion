// Loading fem-result/1 files inside Remotion: files live under public/ (resolved with staticFile), loading holds the
// render with delayRender, and each file is fetched once per bundle.
import {useEffect, useState} from 'react';
import {continueRender, delayRender, staticFile} from 'remotion';
import {fetchFemResult, type FemData} from '@femotion/core';

const cache = new Map<string, Promise<FemData>>();
/** path: a path under public/ without extension, e.g. 'fem/plate' */
export const loadFemResult = (path: string): Promise<FemData> => {
	if (!cache.has(path)) {
		cache.set(path, fetchFemResult(staticFile(`${path}.json`), staticFile(`${path}.bin`)));
	}
	return cache.get(path)!;
};

export const useFemResult = (path: string): FemData | null => {
	const [d, setD] = useState<FemData | null>(null);
	const [h] = useState(() => delayRender(`loading fem result ${path}`));
	useEffect(() => {
		loadFemResult(path).then((r) => {
			setD(r);
			continueRender(h);
		});
	}, [path, h]);
	return d;
};
